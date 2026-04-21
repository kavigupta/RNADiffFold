# -*- coding: utf-8 -*-
"""
DDPO fine-tuning of RNADiffFold on DMS (Dimethyl Sulfate) chemical probing data.

Uses DDPO-SF (score function / REINFORCE) to optimize the diffusion model's denoising
network for maximizing Pearson correlation between predicted structure and DMS reactivity.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm


def pearson_correlation(pred, target, eps=1e-8):
    """Differentiable Pearson correlation coefficient, handling NaN.

    Args:
        pred: FloatTensor (N,)
        target: FloatTensor (N,) with NaN at missing values

    Returns:
        scalar: Pearson r in [-1, 1], or 0 if insufficient valid data
    """
    mask = ~torch.isnan(target)
    if mask.sum() < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred = pred[mask]
    target = target[mask]

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    cov = (pred_centered * target_centered).mean()
    pred_std = pred_centered.norm()
    target_std = target_centered.norm()

    corr = cov / (pred_std * target_std + eps)
    return corr


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
            # Convert one-hot to sequence string
            seq_str = self._onehot_to_sequence(x)
            self.data.append((x, y, seq_str))

    @staticmethod
    def _onehot_to_sequence(onehot):
        """Convert one-hot tensor to sequence string.

        Args:
            onehot: (L, 4) tensor with A, C, G, T one-hot encoding

        Returns:
            sequence: str of length L
        """
        # onehot uses ACGT ordering (from contiguous_regions)
        alphabet = "ACGT"
        indices = torch.argmax(onehot, dim=-1)  # (L,)
        sequence = "".join(alphabet[idx.item()] for idx in indices)
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, seq_str = self.data[idx]
        # Convert y (numpy) to torch tensor
        y_tensor = torch.from_numpy(y).float()
        return x, y_tensor, seq_str


class DDPOTrainer:
    """DDPO-SF trainer for RNADiffFold."""

    def __init__(
        self,
        model,
        device,
        n_samples=4,
        lr=1e-5,
        kl_weight=0.1,
        accumulation_steps=1
    ):
        """
        Args:
            model: DiffusionRNA2dPrediction instance (loaded checkpoint)
            device: torch device
            n_samples: number of sample trajectories per sequence
            lr: learning rate
            kl_weight: weight for KL regularization (unused for now, reserved for future)
            accumulation_steps: gradient accumulation steps
        """
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.lr = lr
        self.kl_weight = kl_weight
        self.accumulation_steps = accumulation_steps

        # Get FM tokenizer
        self.alphabet = model.get_alphabet()

        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def sequence_to_tokens(self, seq_str):
        """Convert sequence string to RNA-FM token IDs.

        Args:
            seq_str: sequence string (e.g., "ACGTACGT")

        Returns:
            token_ids: LongTensor of token IDs with <cls> and <eos> added
        """
        # Use RNA-FM alphabet for tokenization
        # The alphabet object should have encode method
        if hasattr(self.alphabet, 'encode'):
            return self.alphabet.encode(seq_str)
        else:
            # Fallback: manually encode
            # RNA-FM typically: <pad>=0, <unk>=1, <cls>=2, <eos>=3, A=4, C=5, G=6, U=7
            token_map = {'A': 4, 'C': 5, 'G': 6, 'U': 7, 'T': 7}  # T maps to U
            tokens = [2]  # <cls>
            tokens.extend(token_map.get(c, 1) for c in seq_str.upper())
            tokens.append(3)  # <eos>
            return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def compute_reward(self, model_prob, dms_vals, center_start=80, center_end=160):
        """Compute correlation reward between predicted structure and DMS.

        Args:
            model_prob: (B, 1, L_pad, L_pad) soft paired probabilities
            dms_vals: (B, L_center) DMS reactivity with NaN at missing positions
            center_start: start index of center window (context-aware)
            center_end: end index of center window

        Returns:
            rewards: (B,) scalar per-sequence correlation
        """
        # Marginal: per-position max probability of being paired
        model_prob_2d = model_prob.squeeze(1)  # (B, L_pad, L_pad)
        paired_prob = model_prob_2d.max(dim=-1).values  # (B, L_pad)

        # Extract center window
        paired_prob_center = paired_prob[:, center_start:center_end]  # (B, L_center)

        # Compute correlations per sequence
        batch_size = paired_prob_center.shape[0]
        rewards = []
        for i in range(batch_size):
            # Negative correlation: high paired prob → low DMS → negate to get reward
            r = pearson_correlation(paired_prob_center[i], dms_vals[i])
            rewards.append(-r)  # Negate: high paired prob should give high reward when DMS is low

        return torch.stack(rewards)

    def prepare_data_fcn_2(self, seq_onehot):
        """Compute data_fcn_2 (outer product features) from one-hot sequence.

        Args:
            seq_onehot: (B, L, 4) one-hot tensor (A, U, C, G)

        Returns:
            data_fcn_2: (B, 17, L, L) feature tensor
        """
        batch_size, seq_len, _ = seq_onehot.shape

        # Create outer product channels (16 channels from 4×4 combinations)
        outer_products = []
        for i in range(4):
            for j in range(4):
                prod = torch.einsum("bl,bm->blm", seq_onehot[:, :, i], seq_onehot[:, :, j])
                outer_products.append(prod)

        outer_stack = torch.stack(outer_products, dim=1)  # (B, 16, L, L)

        # Add creatmat-like channel (simplified: diagonal preference)
        creatmat = torch.zeros((batch_size, 1, seq_len, seq_len), device=seq_onehot.device, dtype=seq_onehot.dtype)
        # Simple heuristic: GC and AU base pairs
        gc_au_score = seq_onehot[:, :, 2:4].sum(dim=2)  # (B, L) - C and G
        au_score = seq_onehot[:, :, 0:2].sum(dim=2)  # (B, L) - A and U
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    # Simple scoring: give higher weight to canonical pairs
                    score = (gc_au_score[b, i] * gc_au_score[b, j] + au_score[b, i] * au_score[b, j]) / 2.0
                    creatmat[b, 0, i, j] = score
                    creatmat[b, 0, j, i] = score

        data_fcn_2 = torch.cat([outer_stack, creatmat], dim=1)  # (B, 17, L, L)
        return data_fcn_2

    def ddpo_step(self, batch_x, batch_dms, batch_seq_strs):
        """Single DDPO training step.

        Args:
            batch_x: (B, L_pad, 4) one-hot sequences
            batch_dms: (B, L_center) DMS reactivity values
            batch_seq_strs: list of B sequence strings

        Returns:
            loss: scalar loss
            rewards_mean: mean reward (for logging)
        """
        batch_x = batch_x.to(self.device)
        batch_dms = batch_dms.to(self.device)

        batch_size = batch_x.shape[0]
        set_max_len = batch_x.shape[1]

        # Prepare conditioning inputs
        data_fcn_2 = self.prepare_data_fcn_2(batch_x).to(self.device)

        # Tokenize sequences for FM model
        # Pad to max length in batch
        tokenized = [self.sequence_to_tokens(seq) for seq in batch_seq_strs]
        max_token_len = max(len(t) for t in tokenized)
        data_seq_raw = torch.zeros((batch_size, max_token_len), dtype=torch.long, device=self.device)
        for i, tokens in enumerate(tokenized):
            data_seq_raw[i, :len(tokens)] = tokens.to(self.device)

        seq_encoding = batch_x
        contact_masks = torch.ones((batch_size, 1, set_max_len, set_max_len), device=self.device)

        # Sample multiple trajectories
        all_rewards = []
        all_log_probs = []

        for sample_idx in range(self.n_samples):
            # Sample trajectory with log probs
            pred_x0, model_prob, trajectory_log_probs = self.model.sample_with_log_probs(
                batch_size,
                data_fcn_2,
                data_seq_raw,
                set_max_len,
                contact_masks,
                seq_encoding,
                do_pbar=False
            )

            # Compute rewards
            rewards = self.compute_reward(model_prob, batch_dms)
            all_rewards.append(rewards)
            all_log_probs.append(trajectory_log_probs)

        # Stack: (n_samples, B)
        rewards = torch.stack(all_rewards)  # (n_samples, B)
        log_probs = torch.stack(all_log_probs)  # (n_samples, B)

        # Advantage normalization
        advantages = rewards - rewards.mean(dim=0, keepdim=True)
        advantages = advantages / (advantages.std(dim=0, keepdim=True) + 1e-8)

        # DDPO-SF loss: negative policy gradient
        pg_loss = -(log_probs * advantages.detach()).mean()

        return pg_loss, rewards.mean().item()

    def train(self, train_loader, n_epochs, checkpoint_dir="./RNADiffFold/ckpt/model_ckpt"):
        """Train for n_epochs.

        Args:
            train_loader: DataLoader yielding (x_batch, dms_batch, seq_strs)
            n_epochs: number of epochs
            checkpoint_dir: where to save checkpoints
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.model.train()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_rewards = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                # Unpack batch (handle both 2-tuple and 3-tuple formats)
                if len(batch) == 3:
                    batch_x, batch_dms, batch_seq_strs = batch
                else:
                    batch_x, batch_dms = batch
                    # Fallback: reconstruct sequence strings from one-hot
                    batch_seq_strs = [DDPODataset._onehot_to_sequence(x) for x in batch_x]

                loss, reward = self.ddpo_step(batch_x, batch_dms, batch_seq_strs)

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

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_path = Path(checkpoint_dir) / f"ddpo_dms_epoch_{epoch+1}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        # Save final checkpoint
        final_ckpt = Path(checkpoint_dir) / "ddpo_dms.pt"
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"Training complete. Final checkpoint: {final_ckpt}")
