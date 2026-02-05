#!/usr/bin/env python
"""
Test that predict_from_onehot(onehot, model, "AUCG") matches the prediction.py pipeline
on the same FASTA data. Run from RNADiffFold or heals-dms with
  python RNADiffFold/prediction/test_predict_from_onehot.py [--quick]
  python -m prediction.test_predict_from_onehot [--quick]
--quick: use num_samples=1 and only the first sequence (faster). With --quick, Path B
  uses a smaller set_max_len (from one seq only), so results can differ; run without
  --quick to verify equality (all sequences, same set_max_len).
"""
import argparse
import os
import sys
from os.path import abspath, dirname, join

import numpy as np
import torch

# Run from repo root (heals-dms) or RNADiffFold: ensure prediction module is importable
_ROOT = dirname(abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PARENT = dirname(_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import predict_from_onehot as pfo
from prediction_utils import (
    get_data,
    get_data_from_onehot,
    get_model_prediction,
    padding,
    process_config,
    seq2encoding,
    set_seed,
)

from prediction import prediction


def main():
    parser = argparse.ArgumentParser(
        description="Test predict_from_onehot vs prediction.py"
    )
    parser.add_argument(
        "--quick", action="store_true", help="num_samples=1, first sequence only"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override num_samples (e.g. 2 for faster full run)",
    )
    args = parser.parse_args()

    # Use prediction/ as root so paths match prediction.py
    root = dirname(abspath(__file__))
    os.chdir(root)
    parent = dirname(root)

    config = process_config(join(root, "config.json"))
    set_seed(config.seed)

    print("Loading model...")
    model, alphabet = get_model_prediction(config.model)
    ckpt_path = config.model_ckpt_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = join(root, ckpt_path)
    if not os.path.isfile(ckpt_path):
        ckpt_path = join(parent, "ckpt", "model_ckpt", "train.seed.2023.pt")
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print(
            "Skipping comparison; building onehot from FASTA and running predict_from_onehot only."
        )
        load_ckpt = False
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        load_ckpt = True

    fasta_path = join(root, "predict_data", config.predict_data)
    if not os.path.isfile(fasta_path):
        print(f"FASTA not found: {fasta_path}")
        return 1

    # Load FASTA once (same as prediction.py)
    set_seed(config.seed)
    (
        data_fcn_2,
        tokens,
        seq_encoding_pad,
        seq_length,
        name_list,
        set_max_len,
        seq_list,
        seq_len_list,
    ) = get_data(fasta_path, alphabet)
    if args.quick:
        data_fcn_2 = data_fcn_2[:1]
        tokens = tokens[:1]
        seq_encoding_pad = seq_encoding_pad[:1]
        seq_length = seq_length[:1]
        name_list = name_list[:1]
        seq_list = seq_list[:1]
        seq_len_list = seq_len_list[:1]

    run_config = type("Config", (), {})()
    run_config.device = config.device if hasattr(config, "device") else 0
    run_config.num_samples = (
        args.samples
        if args.samples is not None
        else (1 if args.quick else getattr(config, "num_samples", 10))
    )
    run_config.seed = getattr(config, "seed", 2023)

    # ---- Path A: prediction() on get_data outputs (same as prediction.py) ----
    if load_ckpt:
        print("Path A: prediction() on get_data outputs...")
        set_seed(config.seed)
        pred_list_A = prediction(
            run_config,
            model,
            data_fcn_2,
            tokens,
            seq_encoding_pad,
            seq_length,
            set_max_len,
        )
        result_A = np.stack([p.cpu().numpy() for p in pred_list_A], axis=0)
    else:
        result_A = None

    print(result_A.shape)
    print(result_A.mean())
    print(result_A.sum())

    # ---- Path B: onehot from same seq_list (A,U,C,G order) + predict_from_onehot ----
    print("Path B: onehot from FASTA seqs (order=AUCG) + predict_from_onehot()...")
    set_seed(config.seed)
    onehot_list = [padding(seq2encoding(seq), set_max_len) for seq in seq_list]
    onehot_np = np.stack(onehot_list, axis=0)
    onehot = torch.tensor(onehot_np, dtype=torch.float32)

    print(onehot.cpu().numpy().tolist())

    result_B = pfo.predict_from_onehot(
        onehot, model, "AUCG", num_samples=run_config.num_samples, seed=run_config.seed
    )
    print(result_B.mean())
    result_B = result_B.cpu().numpy()

    # ---- Sanity: decoded sequence matches FASTA ----
    decoded = pfo.onehot_to_seq(onehot, "AUCG")
    if len(seq_list) == 1:
        decoded = [decoded]
    for i, (name, seq, dec) in enumerate(zip(name_list, seq_list, decoded)):
        if seq != dec:
            print(
                f"  WARNING: decoded seq {i} ({name}) != FASTA seq:\n    FASTA: {seq[:60]}...\n    decoded: {dec[:60]}..."
            )
        else:
            print(f"  Sequence {i} ({name}): decoded matches FASTA (len={len(seq)})")

    # ---- Compare (valid region only; set_max_len can differ between paths) ----
    if result_A is not None:
        seq_lens = seq_length.cpu().numpy()
        print(f"Path A (get_data + prediction): shape {result_A.shape}")
        print(f"Path B (predict_from_onehot):  shape {result_B.shape}")
        max_diff_all = 0
        n_diff_all = 0
        for b in range(len(seq_lens)):
            slen = int(seq_lens[b])
            block_A = result_A[b, :slen, :slen]
            block_B = result_B[b, :slen, :slen]
            if block_B.shape != block_A.shape:
                print(
                    f"  Sample {b}: valid shape mismatch A {block_A.shape} vs B {block_B.shape}"
                )
                return 1
            diff = np.abs(block_A - block_B)
            max_diff_all = max(max_diff_all, diff.max())
            n_diff_all += int(np.sum(diff > 0.5))
        print(
            f"Valid region comparison: max |A - B| = {max_diff_all}; number of (i,j) differing = {n_diff_all}"
        )
        if max_diff_all < 0.01 and n_diff_all == 0:
            print(
                "PASS: predict_from_onehot matches prediction.py pipeline (valid region)."
            )
            return 0
        else:
            print("FAIL: results differ in valid region.")
            if args.quick:
                print(
                    "  (With --quick, set_max_len differs between paths; run without --quick for equality.)"
                )
            return 1
    else:
        # Just report stats for Path B
        B, L, _ = result_B.shape
        for b in range(B):
            slen = int(seq_length[b].item())
            block = result_B[b, :slen, :slen]
            n_one = int(np.sum(block > 0.5))
            n_total = slen * slen
            print(
                f"  Sample {b} ({name_list[b]}): valid region {slen}x{slen}, entries=1: {n_one}/{n_total} ({100*n_one/n_total:.1f}%)"
            )
        print("(Checkpoint not loaded; comparison skipped.)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
