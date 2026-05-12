# -*- coding: utf-8 -*-
"""
DDPO fine-tuning of RNADiffFold on DMS (Dimethyl Sulfate) chemical probing data.

Uses DDPO-SF (score function / REINFORCE) to optimize the diffusion model's denoising
network for maximizing Pearson correlation between predicted structure and DMS reactivity.
"""

import copy

import torch
from pathlib import Path
from tqdm import tqdm

from prediction.predict_from_onehot import _onehot_to_model_order
from prediction.prediction_utils import get_data_from_onehot


def pearson_correlation_batched(pred, target, eps=1e-8):
    """Differentiable per-row Pearson correlation, vectorized across the batch,
    handling NaN in `target` per row.

    Args:
        pred:   FloatTensor (B, N)
        target: FloatTensor (B, N), NaN at missing positions (per row)

    Returns:
        FloatTensor (B,): Pearson r per row; rows with <2 valid entries return 0.
    """
    mask = ~torch.isnan(target)                                    # (B, N)
    n_valid = mask.sum(dim=1)                                      # (B,)
    denom = n_valid.clamp(min=1).to(pred.dtype)                    # avoid 0-div

    target_clean = torch.where(mask, target, torch.zeros_like(target))
    pred_masked = pred * mask                                      # zero invalid

    pred_mean = pred_masked.sum(dim=1) / denom                     # (B,)
    target_mean = target_clean.sum(dim=1) / denom                  # (B,)

    pred_centered = (pred - pred_mean.unsqueeze(1)) * mask
    target_centered = (target_clean - target_mean.unsqueeze(1)) * mask

    cov = (pred_centered * target_centered).sum(dim=1) / denom
    pred_norm = pred_centered.norm(dim=1)
    target_norm = target_centered.norm(dim=1)
    corr = cov / (pred_norm * target_norm + eps)

    # Match the unbatched contract: insufficient valid data -> 0.
    enough = n_valid >= 2
    return torch.where(enough, corr, torch.zeros_like(corr))


class DDPODataset(torch.utils.data.Dataset):
    """Dataset wrapper around contiguous_regions() from accessibility_vs_folding.py."""

    def __init__(self, contiguous_regions_iter):
        """
        Args:
            contiguous_regions_iter: iterator from contiguous_regions(chrom, cano)
                yields (x, y) where:
                x: (L*3, 4) one-hot tensor
                y: (L,) numpy array with np.nan at missing positions
        """
        self.data = []
        for x, y in contiguous_regions_iter:
            # contiguous_regions emits ACGT ordering; the model expects AUCG.
            x = _onehot_to_model_order(x, "ACGT")
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        y_tensor = torch.from_numpy(y).float()
        return x, y_tensor


class DDPOTrainer:
    """DDPO-SF trainer for RNADiffFold."""

    def __init__(
        self,
        model,
        device,
        n_samples=4,
        lr=1e-5,
        kl_weight=0.1,
    ):
        """
        Args:
            model: DiffusionRNA2dPrediction instance (loaded checkpoint)
            device: torch device
            n_samples: number of sample trajectories per sequence
            lr: learning rate
            kl_weight: weight of per-step KL(policy || frozen reference) anchor
                added to the policy-gradient loss. The reference is a frozen
                snapshot of `model` taken at construction time; without it,
                DDPO reliably collapses to degenerate high-reward modes.
        """
        # Advantage normalization in ddpo_step divides by the std across the
        # n_samples axis; with a single sample that std is zero (NaN under
        # unbiased estimator) and the policy gradient blows up. Require >1.
        assert n_samples > 1, (
            f"n_samples must be > 1 for advantage normalization, got {n_samples}"
        )

        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.lr = lr
        self.kl_weight = kl_weight

        # Get FM tokenizer
        self.alphabet = model.get_alphabet()

        # Restrict DDPO updates to the denoiser. The U-Fold encoder is loaded
        # with requires_grad=True (see models/model.py:load_u_conditioner) for
        # joint training under the supervised objective in train.py, but here
        # the only learning signal is a noisy REINFORCE estimate of a sparse
        # per-trajectory reward. Updating a large pretrained conditioning
        # stack under that signal tends to corrupt the conditioning features
        # before the policy converges, so we freeze it -- matching how
        # fm_conditioner is already treated (its forward is wrapped in
        # torch.no_grad in get_fm_embedding).
        model.u_conditioner.requires_grad_(False)
        model.u_conditioner.eval()

        # FM is meant to be a frozen feature extractor: its forward in
        # get_fm_embedding runs under torch.no_grad. Assert the invariants
        # rather than enforcing them here, so a future change to model.py
        # (e.g. unfreezing FM for joint training) fails loudly instead of
        # being silently re-frozen by DDPO init.
        assert not model.fm_conditioner.training, "FM conditioner must be in eval mode for DDPO"
        assert all(not p.requires_grad for p in model.fm_conditioner.parameters()), (
            "FM conditioner parameters must be frozen for DDPO"
        )

        # Frozen reference policy for the KL anchor. Deep-copied so subsequent
        # optimizer steps on `model` do not drift the reference. The FM
        # conditioner is frozen in both copies (always run under no_grad), so
        # we share it to halve the FM memory footprint.
        self.ref_model = copy.deepcopy(model)
        self.ref_model.fm_conditioner = model.fm_conditioner
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr
        )
        self.scheduler = None  # created in train() once n_epochs is known

    def compute_reward(self, model_prob, dms_vals, full_len):
        """Compute correlation reward between predicted structure and DMS.

        Args:
            model_prob: (B, 1, L_pad, L_pad) soft paired probabilities
            dms_vals: (B, L_center) DMS reactivity with NaN at missing positions
            full_len: unpadded sequence length L (heals-dms emits regions of
                length 3*L_center with DMS reported only on the center third).

        Returns:
            rewards: (B,) scalar per-sequence correlation
        """
        # heals-dms convention: each region has length 3 * L_center, with DMS
        # reactivity reported only on the center third [L_center, 2*L_center).
        # Derive the window from full_len instead of hardcoding so the trainer
        # adapts if regions are emitted at a different size.
        assert full_len % 3 == 0, (
            f"full_len {full_len} is not divisible by 3; cannot place center window"
        )
        center_len = full_len // 3
        center_start = center_len
        center_end = 2 * center_len
        assert dms_vals.shape[1] == center_len, (
            f"dms_vals length {dms_vals.shape[1]} does not match center window "
            f"[{center_start}:{center_end}] (size {center_len})"
        )

        # Per-position P(paired with anything) under independent-cell semantics:
        # the differentiable analog of `result.any(2)` used by the eval pipeline
        # (healsdms accessibility_vs_folding.compute_rna_diff_fold). Compared to
        # `max(dim=-1)`, prob-OR distributes the gradient across all candidate
        # partners instead of just the single argmax column.
        model_prob_2d = model_prob.squeeze(1).clamp(0.0, 1.0)  # (B, L_pad, L_pad)
        log_unpaired = torch.log1p(-model_prob_2d + 1e-8).sum(dim=-1)  # (B, L_pad)
        paired_prob = 1.0 - torch.exp(log_unpaired)

        # Extract center window
        paired_prob_center = paired_prob[:, center_start:center_end]  # (B, L_center)

        # We want a negative correlation, since high DMS = low pairing.
        return -pearson_correlation_batched(paired_prob_center, dms_vals)

    def ddpo_step(self, batch_x, batch_dms):
        """Single DDPO training step.

        Args:
            batch_x: (B, L, 4) one-hot sequences in AUCG order (full length, no end-padding)
            batch_dms: (B, L_center) DMS reactivity values

        Returns:
            loss: scalar loss
            rewards_mean: mean reward (for logging)
        """
        batch_x = batch_x.to(self.device)
        batch_dms = batch_dms.to(self.device)

        batch_size, full_len, _ = batch_x.shape
        seq_lengths = torch.full((batch_size,), full_len, dtype=torch.long, device=batch_x.device)

        data_fcn_2, data_seq_raw, seq_encoding, _, set_max_len = get_data_from_onehot(
            batch_x, self.alphabet, seq_lengths=seq_lengths
        )
        data_fcn_2 = data_fcn_2.to(self.device)
        data_seq_raw = data_seq_raw.to(self.device)
        seq_encoding = seq_encoding.to(self.device)

        # The all-ones contact mask below is only correct when no padding region
        # exists. get_data_from_onehot may round set_max_len up past full_len;
        # if so the model would compute log-probs / KL over padding positions
        # and the reward's row-reduction would mix in padded columns. Fail
        # loudly instead of silently degrading.
        assert set_max_len == full_len, (
            f"set_max_len ({set_max_len}) != full_len ({full_len}); "
            "build a real contact_masks from seq_lengths before relaxing this."
        )
        contact_masks = torch.ones((batch_size, 1, set_max_len, set_max_len), device=self.device)

        # Sample multiple trajectories
        all_rewards = []
        all_log_probs = []
        all_kls = []

        for sample_idx in range(self.n_samples):
            pred_x0, model_prob, trajectory_log_probs, trajectory_kl = self.model.sample_with_log_probs(
                batch_size,
                data_fcn_2,
                data_seq_raw,
                set_max_len,
                contact_masks,
                seq_encoding,
                self.ref_model,
                do_pbar=False,
            )

            # Compute rewards
            rewards = self.compute_reward(model_prob, batch_dms, full_len)
            all_rewards.append(rewards)
            all_log_probs.append(trajectory_log_probs)
            all_kls.append(trajectory_kl)

        # Stack: (n_samples, B)
        rewards = torch.stack(all_rewards)  # (n_samples, B)
        log_probs = torch.stack(all_log_probs)  # (n_samples, B)
        kls = torch.stack(all_kls)  # (n_samples, B)

        # Advantage normalization
        advantages = rewards - rewards.mean(dim=0, keepdim=True)
        advantages = advantages / (advantages.std(dim=0, keepdim=True) + 1e-8)

        # DDPO-SF (score-function) loss: gradient flows only through
        # `log_probs` (the differentiable trajectory log-probability accumulated
        # in sample_with_log_probs). `rewards` is computed from `model_prob`,
        # which still carries a pathwise grad through the final step's logits,
        # so we must `.detach()` advantages to keep this a pure REINFORCE
        # estimator. See log_sample_categorical / p_sample_with_grad in
        # models/diffusion_multinomial.py for the gradient semantics.
        pg_loss = -(log_probs * advantages.detach()).mean()

        # KL anchor to the frozen reference policy. Gradient flows through
        # the current policy's per-step log_probs (the reference is no-grad
        # inside sample_with_log_probs).
        loss = pg_loss + self.kl_weight * kls.mean()

        return loss, rewards.mean().item()

    def train(self, train_loader, n_epochs, checkpoint_dir="./ckpt/model_ckpt"):
        """Train for n_epochs.

        Args:
            train_loader: DataLoader yielding (x_batch, dms_batch)
            n_epochs: number of epochs
            checkpoint_dir: where to save checkpoints
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs
        )

        self.model.train()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_rewards = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                batch_x, batch_dms = batch
                loss, reward = self.ddpo_step(batch_x, batch_dms)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_rewards += reward
                n_batches += 1

            self.scheduler.step()

            avg_loss = epoch_loss / n_batches
            avg_reward = epoch_rewards / n_batches

            print(
                f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Reward: {avg_reward:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            ckpt_path = Path(checkpoint_dir) / f"{epoch+1}.pt"
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        print(f"Training complete. Checkpoints in: {checkpoint_dir}")
