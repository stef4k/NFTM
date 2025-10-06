"""CNN baseline trainer for the 2-D heat equation."""
import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_eq import generate_sequence


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    size: int = 32
    timesteps: int = 10
    batch_size: int = 16
    train_steps: int = 5000
    eval_batches: int = 16
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    alpha_min: float = 0.05
    alpha_max: float = 0.2
    rollout_weight: float = 0.0
    channels: int = 64
    depth: int = 6

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(x + out)


class CNNResidualStep(nn.Module):
    """Residual CNN that predicts the next heat field."""

    def __init__(self, channels: int = 64, depth: int = 6):
        super().__init__()
        self.in_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(depth)])
        self.out_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in_conv(field))
        for block in self.blocks:
            x = block(x)
        delta = self.out_conv(x)
        return field + delta

    def rollout(self, field: torch.Tensor, timesteps: int) -> torch.Tensor:
        seq = [field]
        u = field
        for _ in range(timesteps - 1):
            u = self.forward(u)
            seq.append(u)
        return torch.stack(seq, dim=1)


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def sample_heat_batch(cfg: TrainConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = random.uniform(cfg.alpha_min, cfg.alpha_max)
    seq = generate_sequence(cfg.batch_size, cfg.size, cfg.size, cfg.timesteps, alpha, device)
    inputs = seq[:, :-1].unsqueeze(2)  # B × (T-1) × 1 × H × W
    targets = seq[:, 1:].unsqueeze(2)
    B, T, C, H, W = inputs.shape
    inputs = inputs.reshape(B * T, C, H, W)
    targets = targets.reshape(B * T, C, H, W)
    return inputs, targets


def compute_rollout_loss(model: CNNResidualStep, seq0: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute rollout loss comparing entire sequence."""
    rollout = model.rollout(seq0, target.shape[1])
    return F.mse_loss(rollout, target)


def train(model: CNNResidualStep, cfg: TrainConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    running_loss = 0.0
    running_count = 0
    for step in range(1, cfg.train_steps + 1):
        model.train()
        inputs, targets = sample_heat_batch(cfg, device)
        preds = model(inputs)
        loss = F.mse_loss(preds, targets)

        if cfg.rollout_weight > 0:
            seq_inputs = inputs.reshape(cfg.batch_size, cfg.timesteps - 1, 1, cfg.size, cfg.size)
            seq_targets = targets.reshape(cfg.batch_size, cfg.timesteps - 1, 1, cfg.size, cfg.size)
            seq = torch.cat([seq_inputs[:, :1], seq_targets], dim=1)
            rollout_loss = compute_rollout_loss(model, seq[:, 0], seq)
            loss = loss + cfg.rollout_weight * rollout_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        running_count += 1
        if step % 100 == 0 or step == cfg.train_steps:
            window = running_count
            avg_loss = running_loss / window
            print(f"Step {step:05d}: loss={avg_loss:.6f}")
            running_loss = 0.0
            running_count = 0

    metrics = evaluate(model, cfg)
    metrics["train_steps"] = cfg.train_steps
    return metrics


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def evaluate(model: CNNResidualStep, cfg: TrainConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.eval()

    total_mse = 0.0
    rollout_mse = 0.0
    rollout_elems = 0
    num_samples = 0

    example = None
    for _ in range(cfg.eval_batches):
        alpha = random.uniform(cfg.alpha_min, cfg.alpha_max)
        seq = generate_sequence(cfg.batch_size, cfg.size, cfg.size, cfg.timesteps, alpha, device)
        inputs = seq[:, :-1].unsqueeze(2)
        targets = seq[:, 1:].unsqueeze(2)
        B, T, C, H, W = inputs.shape

        inputs_flat = inputs.reshape(B * T, C, H, W)
        targets_flat = targets.reshape(B * T, C, H, W)
        preds = model(inputs_flat)

        mse_step = F.mse_loss(preds, targets_flat, reduction="sum").item()
        total_mse += mse_step
        num_samples += preds.numel()

        rollout_pred = model.rollout(seq[:, 0].unsqueeze(1), cfg.timesteps)
        rollout_true = seq.unsqueeze(2)
        rollout_mse += F.mse_loss(rollout_pred, rollout_true, reduction="sum").item()
        rollout_elems += rollout_pred.numel()

        if example is None:
            example = {
                "pred": rollout_pred[0].cpu(),
                "true": rollout_true[0, :, 0].cpu(),
            }

    total_mse /= num_samples
    rollout_mse /= rollout_elems

    metrics = {
        "teacher_forced_mse": float(total_mse),
        "teacher_forced_psnr": psnr(total_mse),
        "rollout_mse": float(rollout_mse),
        "rollout_psnr": psnr(rollout_mse),
    }

    if example is not None:
        save_rollout_figure(example["pred"], example["true"], cfg)

    return metrics


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def save_rollout_figure(pred: torch.Tensor, true: torch.Tensor, cfg: TrainConfig) -> None:
    out_dir = Path(f"out/cnn_{cfg.size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    timesteps = pred.shape[0]
    cols = min(5, timesteps)
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, None]
    for i in range(cols):
        axes[0, i].imshow(true[i].squeeze().numpy(), cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"GT t={i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(pred[i].squeeze().numpy(), cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1, i].set_title(f"Pred t={i}")
        axes[1, i].axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "rollout.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a residual CNN baseline for the heat equation")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.2)
    parser.add_argument("--rollout_weight", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()

    cfg = TrainConfig(
        size=args.size,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        rollout_weight=args.rollout_weight,
        channels=args.channels,
        depth=args.depth,
    )

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    model = CNNResidualStep(channels=cfg.channels, depth=cfg.depth).to(device)

    metrics = train(model, cfg)
    metrics.update(cfg.to_dict())

    out_dir = Path(f"out/cnn_{cfg.size}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "metrics": metrics,
    }, out_dir / "ckpt.pt")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
