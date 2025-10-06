"""PINN-α baseline for the 2D heat equation.

This script learns either a global (scalar) diffusion coefficient α or a
spatially-varying map α(x, y) by minimizing a combination of a supervised
finite-difference step loss and a physics residual loss. The data are generated
on-the-fly from the explicit heat-equation simulator used throughout the NFTM
project. After training, the learned parameter is evaluated by rolling out a
sequence and comparing it against the ground-truth simulator using PSNR.

Example usage:
    python -m baselines.heat_eq.pinn_alpha --size 32 --timesteps 16 --epochs 400
    python -m baselines.heat_eq.pinn_alpha --size 32 --timesteps 16 --spatial_alpha \
        --lambda_pde 1.0 --epochs 800
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from heat_eq import (
    generate_variable_sequence,
    heat_step as simulator_heat_step,
    heat_step_variable as simulator_heat_step_variable,
)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

TRUE_ALPHA = 0.1
DATA_RANGE = 1.0  # random initial conditions lie in [0, 1]


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_softplus(x: float) -> float:
    """Numerically stable inverse for softplus for scalar initialisation."""
    if x <= 0:
        raise ValueError("softplus inverse only defined for positive x")
    return math.log(math.exp(x) - 1.0)


def laplacian(u: torch.Tensor) -> torch.Tensor:
    """5-point stencil discrete Laplacian with Neumann padding."""
    return simulator_heat_step(u, 1.0) - u


def evolve_heat(u: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Wrapper around the repository heat simulator for differentiable steps."""
    if alpha.ndim == 0:
        return simulator_heat_step(u, alpha)
    return simulator_heat_step_variable(u, alpha)


def make_alpha_map(size: int, device: torch.device) -> torch.Tensor:
    """Construct the canonical spatial diffusion map used in the repo."""
    _, alpha_map = generate_variable_sequence(1, size, size, 2, device)
    return alpha_map


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return float("inf")
    return float(20.0 * math.log10(DATA_RANGE) - 10.0 * math.log10(mse.item()))


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class PINNAlpha(nn.Module):
    """Minimal parameterisation for α (scalar or spatial)."""

    def __init__(self, size: int, spatial: bool, init: float = 0.05):
        super().__init__()
        self.spatial = spatial
        init_val = inverse_softplus(init)
        if spatial:
            param = torch.full((size, size), init_val)
        else:
            param = torch.tensor(init_val)
        self.raw_alpha = nn.Parameter(param)

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)


# -----------------------------------------------------------------------------
# Training components
# -----------------------------------------------------------------------------


@dataclass
class Batch:
    u_t: torch.Tensor
    u_tp1: torch.Tensor


def sample_batch(
    batch_size: int,
    size: int,
    timesteps: int,
    device: torch.device,
    spatial: bool,
    true_alpha_map: torch.Tensor,
) -> Batch:
    """Generate a batch of (u_t, u_{t+1}) pairs using the ground truth α."""
    u0 = torch.rand(batch_size, size, size, device=device)
    seq = [u0]
    alpha = true_alpha_map if spatial else torch.tensor(TRUE_ALPHA, device=device)
    for _ in range(timesteps - 1):
        seq.append(evolve_heat(seq[-1], alpha))
    seq = torch.stack(seq, dim=1)
    idx = torch.randint(0, timesteps - 1, (batch_size,), device=device)
    gather = torch.arange(batch_size, device=device)
    u_t = seq[gather, idx]
    u_tp1 = seq[gather, idx + 1]
    return Batch(u_t=u_t, u_tp1=u_tp1)


def train(
    model: PINNAlpha,
    args: argparse.Namespace,
    device: torch.device,
    spatial: bool,
    true_alpha_map: torch.Tensor,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        batch = sample_batch(
            batch_size=args.batch_size,
            size=args.size,
            timesteps=args.timesteps,
            device=device,
            spatial=spatial,
            true_alpha_map=true_alpha_map,
        )
        u_t = batch.u_t
        u_tp1 = batch.u_tp1

        alpha = model()

        lap = laplacian(u_t)
        pred = evolve_heat(u_t, alpha)
        residual = u_tp1 - u_t - alpha * lap

        step_loss = F.mse_loss(pred, u_tp1)
        residual_loss = residual.pow(2).mean()
        loss = args.lambda_sup * step_loss + args.lambda_pde * residual_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            alpha_detached = alpha.detach()
            if spatial:
                alpha_stats = {
                    "min": float(alpha_detached.min().item()),
                    "max": float(alpha_detached.max().item()),
                    "mean": float(alpha_detached.mean().item()),
                }
            else:
                alpha_stats = {"scalar": float(alpha_detached.item())}
            print(
                f"Epoch {epoch:04d} | step_loss={step_loss.item():.6f} "
                f"residual_loss={residual_loss.item():.6f} | α={alpha_stats}"
            )


# -----------------------------------------------------------------------------
# Evaluation / visualisation
# -----------------------------------------------------------------------------


def rollout(
    u0: torch.Tensor,
    steps: int,
    alpha: torch.Tensor,
) -> torch.Tensor:
    seq = [u0]
    u = u0
    for _ in range(steps - 1):
        u = evolve_heat(u, alpha)
        seq.append(u)
    return torch.stack(seq, dim=1)


def render_rollout(
    out_dir: Path,
    gt: torch.Tensor,
    pred: torch.Tensor,
    alpha_true: torch.Tensor,
    alpha_pred: torch.Tensor,
) -> None:
    steps = min(6, gt.shape[1])
    fig, axes = plt.subplots(3, steps, figsize=(3 * steps, 8))
    for t in range(steps):
        ax = axes[0, t]
        im = ax.imshow(gt[0, t].cpu(), cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"GT t={t}")
        ax.axis("off")
        if t == steps - 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, t]
        ax.imshow(pred[0, t].cpu(), cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"Pred t={t}")
        ax.axis("off")

        ax = axes[2, t]
        ax.imshow((gt[0, t] - pred[0, t]).cpu(), cmap="coolwarm")
        ax.set_title("Error")
        ax.axis("off")

    fig.suptitle("PINN-α Rollout Comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "rollout.png", dpi=160)
    plt.close(fig)

    if alpha_true.ndim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(alpha_true.cpu(), cmap="viridis")
        axes[0].set_title("True α")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(alpha_pred.cpu(), cmap="viridis")
        axes[1].set_title("Learned α")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow((alpha_true - alpha_pred).cpu(), cmap="coolwarm")
        axes[2].set_title("α Error")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(out_dir / "alpha_map.png", dpi=160)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PINN baseline for α")
    parser.add_argument("--size", type=int, default=32, help="Spatial resolution")
    parser.add_argument("--timesteps", type=int, default=16, help="Sequence length")
    parser.add_argument(
        "--spatial_alpha",
        action="store_true",
        help="Learn a spatially-varying α map instead of a scalar",
    )
    parser.add_argument("--lambda_pde", type=float, default=1.0, help="Residual weight")
    parser.add_argument("--lambda_sup", type=float, default=1.0, help="Step loss weight")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out_root", type=str, default="out", help="Directory to place outputs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial = args.spatial_alpha

    out_dir = Path(args.out_root) / f"pinn_alpha_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if spatial:
        true_alpha = make_alpha_map(args.size, device)
    else:
        true_alpha = torch.tensor(TRUE_ALPHA, device=device)

    model = PINNAlpha(size=args.size, spatial=spatial).to(device)

    print("== Training PINN-α ==")
    train(model, args, device, spatial, true_alpha)

    model.eval()
    with torch.no_grad():
        learned_alpha = model().detach()

    # Evaluation rollout
    u0 = torch.rand(1, args.size, args.size, device=device)
    gt_rollout = rollout(u0, args.timesteps, true_alpha)
    pred_rollout = rollout(u0, args.timesteps, learned_alpha)

    rollout_psnr = psnr(pred_rollout, gt_rollout)
    print(f"Rollout PSNR: {rollout_psnr:.2f} dB")

    # Persist outputs
    torch.save({"alpha": learned_alpha.cpu()}, out_dir / "ckpt.pt")
    metrics = {
        "psnr": rollout_psnr,
        "alpha_true": float(TRUE_ALPHA) if not spatial else None,
        "alpha_estimate": float(learned_alpha.mean().item())
        if spatial
        else float(learned_alpha.item()),
    }
    if spatial:
        metrics.update({
            "alpha_min": float(learned_alpha.min().item()),
            "alpha_max": float(learned_alpha.max().item()),
        })
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    render_rollout(out_dir, gt_rollout, pred_rollout, true_alpha, learned_alpha)


if __name__ == "__main__":
    main()
