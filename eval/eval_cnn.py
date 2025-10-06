"""Evaluation script for CNN heat-equation baselines."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines.heat_eq.cnn_baseline import CNNResidualStep
from common.heat_metrics import scalar_rollout_psnr, scalar_rollout_ssim
from heat_eq import generate_sequence

matplotlib.use("Agg")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[CNNResidualStep, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {})
    channels = int(config.get("channels", 64))
    depth = int(config.get("depth", 6))
    model = CNNResidualStep(channels=channels, depth=depth)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint {path} does not contain 'model_state_dict'.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    meta = ckpt.get("metrics", {})
    return model, meta


@torch.no_grad()
def rollout_model(model: CNNResidualStep, init_field: torch.Tensor, timesteps: int) -> torch.Tensor:
    """Roll out the CNN model starting from ``init_field``."""
    preds = [init_field]
    current = init_field
    for _ in range(1, timesteps):
        current = model(current)
        preds.append(current)
    return torch.stack(preds, dim=1)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred_seq = pred.unsqueeze(2)
    target_seq = target.unsqueeze(2)
    psnr = scalar_rollout_psnr(pred_seq, target_seq)
    ssim = scalar_rollout_ssim(pred_seq, target_seq)
    return float(psnr), float(ssim)


def save_panel(pred: torch.Tensor, target: torch.Tensor, out_path: Path) -> None:
    """Save a small panel comparing prediction and ground truth."""
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


def extract_train_time(meta: Mapping[str, Any]) -> Any:
    for key in ("train_time_s", "train_time", "train_seconds"):
        if key in meta:
            return meta[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CNN baseline checkpoint on the heat equation")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--size", type=int, required=True, help="Spatial resolution")
    parser.add_argument("--timesteps", type=int, required=True, help="Number of rollout steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for evaluation")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    model, meta = load_checkpoint(args.ckpt, device)

    seq = generate_sequence(args.batch, args.size, args.size, args.timesteps, alpha=0.1, device=device)
    target = seq  # (B, T, H, W)
    init_field = target[:, 0:1]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    pred = rollout_model(model, init_field, args.timesteps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    pred = pred.squeeze(2)
    psnr, ssim = compute_metrics(pred, target)

    total_time = end - start
    step_count = max(args.timesteps - 1, 1)
    inf_time_ms = (total_time / step_count) * 1000.0

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_dir = Path("out") / f"cnn_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "train_time_s": extract_train_time(meta),
        "inf_time_ms": inf_time_ms,
        "params": int(params),
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_panel(pred[0], target[0], out_dir / "samples.png")


if __name__ == "__main__":
    main()
