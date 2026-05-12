#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point for DDPO fine-tuning of RNADiffFold on DMS data.

Usage:
    python ddpo_experiment.py --checkpoint train --chrom chr1 --n_epochs 5 --batch_size 4
"""

import argparse
import itertools
import torch
from pathlib import Path

from healsdms.accessibility_model.accessibility_vs_folding import contiguous_regions

from models.model import DiffusionRNA2dPrediction
from ddpo_finetune import DDPODataset, DDPOTrainer


def load_model_from_checkpoint(checkpoint_path, device="cuda:0"):
    """Load a pre-trained DiffusionRNA2dPrediction model.

    Args:
        checkpoint_path: path to checkpoint .pt file
        device: torch device

    Returns:
        model: DiffusionRNA2dPrediction instance
    """
    model = DiffusionRNA2dPrediction(
        num_classes=2,
        diffusion_dim=8,
        cond_dim=8,
        diffusion_steps=20,  # Use faster inference steps
        dp_rate=0.1,
        u_ckpt="ufold_train_alldata.pt"
    )

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {checkpoint_path}")

    model = model.to(device)
    return model


def load_dms_data(chrom, canonical_bases_only=True, max_samples=None):
    """Load DMS data via the healsdms pipeline.

    Args:
        chrom: chromosome (e.g., "chr1")
        canonical_bases_only: whether to use only canonical bases
        max_samples: maximum number of samples to load (None = all)

    Returns:
        dataset: DDPODataset instance
    """
    contiguous_regions_iter = contiguous_regions(chrom, canonical_bases_only)

    if max_samples:
        contiguous_regions_iter = itertools.islice(contiguous_regions_iter, max_samples)

    dataset = DDPODataset(contiguous_regions_iter)
    print(f"Loaded {len(dataset)} sequences from {chrom}")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="DDPO fine-tuning of RNADiffFold on DMS data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="train",
        help="Base checkpoint to load: 'train' or 'finetune' (in ckpt/model_ckpt/)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Full path to checkpoint (overrides --checkpoint)",
    )
    parser.add_argument(
        "--chrom",
        type=str,
        default="chr1",
        help="Chromosome to train on (e.g., 'chr1', 'chr2')",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (beware memory with large L×L gradient graphs)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of DDPO sample trajectories per sequence",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpt/model_ckpt",
        help="Directory to save checkpoints (run subdirectory created inside)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run identifier; checkpoints land in <output_dir>/<run_id>/<epoch>.pt",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DDPO Fine-tuning of RNADiffFold")
    print("=" * 80)
    print(f"Chromosome: {args.chrom}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"DDPO samples per sequence: {args.n_samples}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load checkpoint
    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
    else:
        ckpt_path = f"./ckpt/model_ckpt/{args.checkpoint}.seed.2023.pt"

    device = torch.device(args.device)
    model = load_model_from_checkpoint(ckpt_path, device=args.device)

    # Load DMS dataset
    dataset = load_dms_data(args.chrom)

    # Create collate function for batch processing
    def collate_ddpo(batch):
        """Collate function for DDPO dataset.

        batch: list of (x, dms) tuples
        Returns: (x_batch, dms_batch)
        """
        x_list, dms_list = zip(*batch)
        x_batch = torch.stack(x_list)
        dms_batch = torch.stack(dms_list)
        return x_batch, dms_batch

    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with torch models
        collate_fn=collate_ddpo,
    )

    # Create trainer
    trainer = DDPOTrainer(
        model,
        device,
        n_samples=args.n_samples,
        lr=args.lr,
    )

    # Train
    run_dir = Path(args.output_dir) / args.run_id
    trainer.train(loader, n_epochs=args.n_epochs, checkpoint_dir=run_dir)

    print("Done!")


if __name__ == "__main__":
    main()
