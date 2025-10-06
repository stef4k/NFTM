import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TrainStats:
    pde_loss: float
    sup_loss: float
    bc_loss: float
    total_loss: float


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding for PINN inputs."""

    def __init__(self, in_dim: int, num_frequencies: int, scale: float = 10.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        if num_frequencies > 0:
            weight = torch.randn(num_frequencies, in_dim) * scale
            self.register_buffer("weight", weight)
        else:
            self.register_buffer("weight", torch.zeros(1, in_dim))

    @property
    def output_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies == 0:
            return torch.empty(coords.shape[0], 0, device=coords.device, dtype=coords.dtype)
        projected = 2.0 * math.pi * coords @ self.weight.t()
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, out_dim: int = 1):
        super().__init__()
        layers = []
        last_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINNModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_fourier: int, fourier_scale: float):
        super().__init__()
        self.fourier = FourierFeatures(3, num_fourier, scale=fourier_scale)
        self.mlp = MLP(self.input_dim, hidden_dim, num_layers)

    @property
    def input_dim(self) -> int:
        return 3 + self.fourier.output_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = torch.cat([coords, self.fourier(coords)], dim=-1)
        return self.mlp(features)


def analytic_solution(coords: torch.Tensor, alpha: float) -> torch.Tensor:
    x, y, t = coords.unbind(-1)
    return (
        torch.sin(math.pi * x)
        * torch.sin(math.pi * y)
        * torch.exp(-2.0 * (math.pi ** 2) * alpha * t)
    ).unsqueeze(-1)


def compute_param_count(in_dim: int, hidden_dim: int, num_layers: int, out_dim: int = 1) -> int:
    if num_layers <= 0:
        return in_dim * out_dim + out_dim
    total = in_dim * hidden_dim + hidden_dim
    for _ in range(num_layers - 1):
        total += hidden_dim * hidden_dim + hidden_dim
    total += hidden_dim * out_dim + out_dim
    return total


def match_hidden_to_target(
    target_params: int,
    in_dim: int,
    num_layers: int,
    out_dim: int = 1,
    min_hidden: int = 16,
    max_hidden: int = 2048,
) -> int:
    if target_params <= 0:
        return min_hidden
    best_hidden = min_hidden
    best_diff = float("inf")
    for hidden in range(min_hidden, max_hidden + 1):
        params = compute_param_count(in_dim, hidden, num_layers, out_dim)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_hidden = hidden
        if params >= target_params and diff <= best_diff:
            break
    return best_hidden


def pde_residual(model: nn.Module, coords: torch.Tensor, alpha: float) -> torch.Tensor:
    coords = coords.clone().detach().requires_grad_(True)
    preds = model(coords)
    grads = torch.autograd.grad(
        preds,
        coords,
        grad_outputs=torch.ones_like(preds),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x, u_y, u_t = grads.unbind(-1)
    u_xx = torch.autograd.grad(
        u_x,
        coords,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y,
        coords,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    residual = u_t.unsqueeze(-1) - alpha * (u_xx + u_yy)
    return residual


def sample_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, 3, device=device)


def sample_boundary(batch_size: int, device: torch.device) -> torch.Tensor:
    coords = torch.rand(batch_size, 3, device=device)
    face = torch.randint(0, 4, (batch_size,), device=device)
    coords[face == 0, 0] = 0.0
    coords[face == 1, 0] = 1.0
    coords[face == 2, 1] = 0.0
    coords[face == 3, 1] = 1.0
    return coords


def evaluate_on_grid(
    model: nn.Module,
    size: int,
    timesteps: int,
    alpha: float,
    device: torch.device,
    chunk: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.0, 1.0, size, device=device)
    ys = torch.linspace(0.0, 1.0, size, device=device)
    ts = torch.linspace(0.0, 1.0, timesteps, device=device)
    preds = []
    with torch.no_grad():
        for t in ts:
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            coord = torch.stack(
                [xx.reshape(-1), yy.reshape(-1), torch.full((size * size,), t, device=device)],
                dim=-1,
            )
            chunk_preds = []
            for start in range(0, coord.shape[0], chunk):
                end = start + chunk
                chunk_preds.append(model(coord[start:end]))
            preds.append(torch.cat(chunk_preds, dim=0))
    pred_grid = torch.stack(preds, dim=0).reshape(timesteps, size, size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gt = []
    for t in ts:
        coords = torch.stack(
            [xx, yy, torch.full_like(xx, t)],
            dim=-1,
        )
        gt.append(analytic_solution(coords.view(-1, 3), alpha).view(size, size))
    gt_grid = torch.stack(gt, dim=0)
    return pred_grid.cpu(), gt_grid.cpu()


def make_figure(pred: torch.Tensor, gt: torch.Tensor, out_path: Path) -> None:
    timesteps = pred.shape[0]
    indices = sorted(set([0, timesteps // 2, timesteps - 1]))
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]
    for row, idx in enumerate(indices):
        axes[row, 0].imshow(gt[idx].numpy(), cmap="viridis", origin="lower")
        axes[row, 0].set_title(f"Ground Truth t={idx}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(pred[idx].numpy(), cmap="viridis", origin="lower")
        axes[row, 1].set_title(f"Prediction t={idx}")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    n_collocation: int,
    n_supervision: int,
    n_boundary: int,
    device: torch.device,
    sup_weight: float,
    bc_weight: float,
) -> TrainStats:
    optimizer.zero_grad()
    coords_collocation = sample_uniform(n_collocation, device)
    residual = pde_residual(model, coords_collocation, alpha)
    loss_pde = (residual ** 2).mean()

    coords_sup = sample_uniform(n_supervision, device)
    targets = analytic_solution(coords_sup, alpha)
    preds_sup = model(coords_sup)
    loss_sup = F.mse_loss(preds_sup, targets)

    coords_bc = sample_boundary(n_boundary, device)
    bc_targets = analytic_solution(coords_bc, alpha)
    preds_bc = model(coords_bc)
    loss_bc = F.mse_loss(preds_bc, bc_targets)

    loss = loss_pde + sup_weight * loss_sup + bc_weight * loss_bc
    loss.backward()
    optimizer.step()
    return TrainStats(
        pde_loss=float(loss_pde.detach().cpu()),
        sup_loss=float(loss_sup.detach().cpu()),
        bc_loss=float(loss_bc.detach().cpu()),
        total_loss=float(loss.detach().cpu()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN baseline with MLP")
    parser.add_argument("--size", type=int, default=32, help="Spatial resolution")
    parser.add_argument("--timesteps", type=int, default=20, help="Number of rollout timesteps")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--param_target", type=int, default=0, help="Target parameter count to match")
    parser.add_argument("--fourier", type=int, default=0, help="Number of Fourier features")
    parser.add_argument("--fourier_scale", type=float, default=10.0, help="Fourier feature scale")
    parser.add_argument("--n_collocation", type=int, default=2048)
    parser.add_argument("--n_supervision", type=int, default=512)
    parser.add_argument("--n_boundary", type=int, default=512)
    parser.add_argument("--sup_weight", type=float, default=1.0)
    parser.add_argument("--bc_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1, help="Diffusion coefficient")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory root")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--steps_per_epoch", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ff_dim = 2 * args.fourier
    in_dim = 3 + ff_dim

    if args.param_target > 0:
        hidden_dim = match_hidden_to_target(args.param_target, in_dim, args.layers)
        print(f"Matched hidden dim {hidden_dim} for target params {args.param_target}")
    else:
        hidden_dim = args.hidden
    print(f"Hidden dimension: {hidden_dim}")

    model = PINNModel(hidden_dim, args.layers, args.fourier, args.fourier_scale).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Using device: {device}")
    print(f"Model parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.out_dir) / f"pinn_mlp_{args.size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_stats = TrainStats(0.0, 0.0, 0.0, 0.0)
        for _ in range(args.steps_per_epoch):
            stats = train_epoch(
                model,
                optimizer,
                args.alpha,
                args.n_collocation,
                args.n_supervision,
                args.n_boundary,
                device,
                args.sup_weight,
                args.bc_weight,
            )
            epoch_stats = TrainStats(
                pde_loss=epoch_stats.pde_loss + stats.pde_loss,
                sup_loss=epoch_stats.sup_loss + stats.sup_loss,
                bc_loss=epoch_stats.bc_loss + stats.bc_loss,
                total_loss=epoch_stats.total_loss + stats.total_loss,
            )
        factor = 1.0 / args.steps_per_epoch
        epoch_stats = TrainStats(
            pde_loss=epoch_stats.pde_loss * factor,
            sup_loss=epoch_stats.sup_loss * factor,
            bc_loss=epoch_stats.bc_loss * factor,
            total_loss=epoch_stats.total_loss * factor,
        )
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | Loss {epoch_stats.total_loss:.6f} | "
                f"PDE {epoch_stats.pde_loss:.6f} | SUP {epoch_stats.sup_loss:.6f} | BC {epoch_stats.bc_loss:.6f}"
            )
    train_time = time.time() - start_time

    model.eval()

    pred_grid, gt_grid = evaluate_on_grid(model, args.size, args.timesteps, args.alpha, device)
    mse = F.mse_loss(pred_grid, gt_grid).item()
    psnr = 10.0 * math.log10(1.0 / (mse + 1e-12))

    metrics = {
        "mse": mse,
        "psnr": psnr,
        "train_time": train_time,
        "params": total_params,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    ckpt_path = output_dir / "ckpt.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)

    fig_path = output_dir / "rollout.png"
    make_figure(pred_grid, gt_grid, fig_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved figure to {fig_path}")
    print(f"Final PSNR: {psnr:.2f} dB | Train time: {train_time:.2f} s")


if __name__ == "__main__":
    main()
