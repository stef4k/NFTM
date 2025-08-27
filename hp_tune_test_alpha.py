#!/usr/bin/env python3
"""
Hyperparameter tuning driver for test_alpha in heat_eq_extended.
Runs a small grid over lr, epochs, and batches per epoch, collects mean abs error,
selects the best configuration, and saves results to JSON.

Usage (defaults are reasonable):
  python hp_tune_test_alpha.py

Optional args:
  --size 64 --timesteps 10 --batch_size 8
  --grid_lr 0.001 0.002 --grid_epochs 120 --grid_batches 1 2 --grid_curr_epochs 30
"""
import argparse
import json
import os
import time
from pathlib import Path

# Use non-interactive backend so plots don't block
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch

import heat_eq as he


def run_one(config):
    # Build an args Namespace compatible with heat_eq_extended
    args = argparse.Namespace(
        mode='test_alpha',
        size=config['size'],
        timesteps=config['timesteps'],
        pe_freqs=8,
        hidden_dim=128,
        batch_size=config['batch_size'],
        epochs_a=200,
        epochs_b=400,
        lr=config['lr'],
        rollout_steps=50,
        train_size=32,
        test_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        test_alpha_epochs=config['test_alpha_epochs'],
        test_alpha_batches_per_epoch=config['test_alpha_batches_per_epoch'],
        test_alpha_curriculum_epochs=config['test_alpha_curriculum_epochs'],
    )

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Run
    learned = he.test_learnable_alpha(args, device)

    # Score: mean absolute error
    keys = sorted(learned.keys())
    mae = float(np.mean([abs(k - float(learned[k])) for k in keys])) if keys else float('inf')

    return mae, learned


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    # Small grid defaults to keep runtime reasonable
    p.add_argument('--grid_lr', type=float, nargs='+', default=[1e-3, 2e-3])
    p.add_argument('--grid_epochs', type=int, nargs='+', default=[120])
    p.add_argument('--grid_batches', type=int, nargs='+', default=[1, 2])
    p.add_argument('--grid_curr_epochs', type=int, nargs='+', default=[30])
    p.add_argument('--out', type=str, default='tune_results_test_alpha.json')
    args = p.parse_args()

    # Build grid
    grid = []
    for lr in args.grid_lr:
        for ep in args.grid_epochs:
            for bpe in args.grid_batches:
                for cep in args.grid_curr_epochs:
                    grid.append({
                        'size': args.size,
                        'timesteps': args.timesteps,
                        'batch_size': args.batch_size,
                        'lr': lr,
                        'test_alpha_epochs': ep,
                        'test_alpha_batches_per_epoch': bpe,
                        'test_alpha_curriculum_epochs': cep,
                    })

    results = []
    best = None
    start = time.time()

    for i, cfg in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] cfg={cfg}")
        t0 = time.time()
        try:
            mae, learned = run_one(cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            mae, learned = float('inf'), {}
        dt = time.time() - t0
        rec = {**cfg, 'mae': mae, 'learned': learned, 'secs': round(dt, 2)}
        results.append(rec)
        if best is None or mae < best['mae']:
            best = rec
        print(f"  -> mae={mae:.4f} (took {dt:.1f}s)")

    total = time.time() - start
    print(f"Done {len(grid)} runs in {total:.1f}s")
    print("Best:", json.dumps(best, indent=2))

    # Save results
    out_path = Path(args.out)
    with out_path.open('w') as f:
        json.dump({'grid': results, 'best': best, 'total_secs': round(total, 2)}, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
