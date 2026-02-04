# -*- coding: utf-8 -*-
"""
Predict RNA secondary structure from batch one-hot encoded sequences (B x L x 4).
"""
import torch

from prediction_utils import (
    get_data_from_onehot,
    set_seed,
)
from prediction import prediction

# Model expects onehot columns in this order: A=0, U=1, C=2, G=3
_MODEL_ORDER = "AUCG"
_DECODE = ("A", "U", "C", "G")  # index -> base in model order


def onehot_to_seq(onehot, order):
    """
    Decode onehot (with channel order `order`) to an RNA sequence string (A/U/C/G).
    Use this to verify the model sees the intended sequence; if the decoded string
    looks wrong, your `order` does not match how the onehot was built.
    """
    reordered = _onehot_to_model_order(
        onehot.cpu() if onehot.is_cuda else onehot, order
    )
    if reordered.dim() == 2:
        reordered = reordered.unsqueeze(0)
    seqs = []
    for b in range(reordered.shape[0]):
        row_sums = reordered[b].abs().sum(dim=-1)
        zeros = (row_sums == 0).nonzero(as_tuple=True)[0]
        length = int(zeros[0].item()) if zeros.numel() else reordered.shape[1]
        idx = reordered[b, :length].argmax(dim=-1)
        seqs.append("".join(_DECODE[i] for i in idx.tolist()))
    return seqs[0] if len(seqs) == 1 else seqs


def _onehot_to_model_order(onehot, order):
    """
    Reorder onehot channels from user `order` to model order (A, U, C, G).
    order: length-4 string, permutation of 'ACGT' (T and U are equivalent).
    """
    order = order.upper()
    if len(order) != 4 or sorted(order.replace("T", "U")) != ["A", "C", "G", "U"]:
        raise ValueError(
            "order must be a length-4 string containing A, C, G, and T (or U) each once, e.g. 'ACGT' or 'TGCA'"
        )
    order_normalized = order.replace("T", "U")
    perm = [order_normalized.index(b) for b in _MODEL_ORDER]
    return onehot[..., perm]


def predict_from_onehot(
    onehot,
    model,
    order,
    *,
    num_samples=10,
    seed=2023,
):
    """
    Run the diffusion model on batch one-hot encoded sequences.

    Parameters
    ----------
    onehot : torch.Tensor, shape (B, L, 4)
        One-hot encoding; column order is given by `order`. Padding rows are all zeros.
    model : DiffusionRNA2dPrediction
        Loaded model.
    order : str, length 4
        Permutation of "ACGT" giving the channel order of `onehot`: order[k] is the base
        for onehot[..., k]. E.g. order="ACGT" means channels are A, C, G, T (T = U in RNA).
    num_samples : int
        Number of samples for voting. Default 10.
    seed : int
        Random seed. Default 2023.

    Returns
    -------
    torch.Tensor, shape (B, set_max_len, set_max_len)
        Predicted contact maps (float). Valid region per sample b is [:seq_len_b, :seq_len_b].
        Values are 0 (unpaired) or 1 (paired). Only a fraction of (i,j) are typically 1;
        if almost all entries in the valid region are 1, the sequence the model sees may be
        wrong—verify that `order` matches your onehot channel order and use
        onehot_to_seq(onehot, order) to check the decoded sequence.
    """
    alphabet = model.get_alphabet()
    device = next(model.parameters()).device

    onehot = _onehot_to_model_order(onehot, order).to(device)

    run_config = type('Config', (), {})()
    run_config.device = device

    run_config.num_samples = num_samples
    run_config.seed = seed

    set_seed(seed)
    data_fcn_2, tokens, seq_encoding_pad, seq_length, set_max_len_out = get_data_from_onehot(
        onehot, alphabet
    )
    best_pred_list = prediction(
        run_config, model, data_fcn_2, tokens, seq_encoding_pad, seq_length, set_max_len_out, do_pbar=False
    )
    return torch.stack([p.to(device) for p in best_pred_list], dim=0)
