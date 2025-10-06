"""Evaluation script for PINN-α checkpoints."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.heat_metrics import scalar_rollout_psnr, scalar_rollout_ssim
from heat_eq import generate_sequence, generate_variable_sequence, heat_step, heat_step_variable

matplotlib.use("Agg")

TRUE_ALPHA = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_alpha(path: Path, device: torch.device) -> torch.Tensor:
    ckpt = torch.load(path, map_location=device)
    alpha = ckpt.get("alpha")
    if alpha is None:
        raise ValueError(f"Checkpoint {path} does not contain an 'alpha' entry.")
    return torch.as_tensor(alpha, dtype=torch.float32, device=device)


def save_panel(pred: torch.Tensor, target: torch.Tensor, out_path: Path) -> None:
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    steps = min(pred_np.shape[0], 6)
    fig, axes = plt.subplots(2, steps, figsize=(3 * steps, 6))
    for idx in range(steps):
        axes[0, idx].imshow(target_np[idx], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(f"GT t={idx}")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(pred_np[idx], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1, idx].set_title(f"Pred t={idx}")
        axes[1, idx].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def rollout_alpha(
    init_field: torch.Tensor,
    timesteps: int,
    alpha: torch.Tensor,
) -> torch.Tensor:
    seq = [init_field]
    current = init_field
    if alpha.ndim == 0:
        alpha_value = float(alpha.item())
        for _ in range(1, timesteps):
            current = heat_step(current, alpha_value)
            seq.append(current)
    else:
        for _ in range(1, timesteps):
            current = heat_step_variable(current, alpha)
            seq.append(current)
    return torch.stack(seq, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PINN-α checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    alpha = load_alpha(args.ckpt, device)
    spatial = alpha.ndim == 2

    if spatial:
        seq, _ = generate_variable_sequence(args.batch, args.size, args.size, args.timesteps, device)
        target = seq
    else:
        seq = generate_sequence(args.batch, args.size, args.size, args.timesteps, TRUE_ALPHA, device)
        target = seq

    init_field = target[:, 0]

    if spatial and alpha.shape != (args.size, args.size):
        raise ValueError(
            f"Alpha map shape {tuple(alpha.shape)} does not match requested size {(args.size, args.size)}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    pred = rollout_alpha(init_field, args.timesteps, alpha)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    psnr = scalar_rollout_psnr(pred.unsqueeze(2), target.unsqueeze(2))
    ssim = scalar_rollout_ssim(pred.unsqueeze(2), target.unsqueeze(2))

    total_time = end - start
    step_count = max(args.timesteps - 1, 1)
    inf_time_ms = (total_time / step_count) * 1000.0

    params = int(alpha.numel())
    alpha_hat = float(alpha.mean().item()) if alpha.ndim > 0 else float(alpha.item())

    out_dir = Path("out") / f"pinn_alpha_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "train_time_s": None,
        "inf_time_ms": inf_time_ms,
        "params": params,
        "alpha_hat": alpha_hat,
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_panel(pred[0], target[0], out_dir / "samples.png")


if __name__ == "__main__":
    main()
