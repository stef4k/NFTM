"""
Extended NFTM Heat Equation Training Script
============================================
Comprehensive script for ML4PS workshop experiments with command-line interface.

Usage examples:
    # Basic training with learnable alpha
    python heat_eq.py --mode learnable_alpha --size 64

    # Variable coefficient experiment (spatially varying alpha)
    python heat_eq.py --mode variable --size 64 --epochs_a 300

    # Baseline comparison (NFTM vs CNN/UNet)
    python heat_eq.py --mode baseline --size 32

    # Test alpha identification routine
    python heat_eq.py --mode test_alpha --size 64

    # Run all primary experiments
    python heat_eq.py --mode all --size 64
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== Ground Truth Simulators ==================

def heat_step(u: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Explicit finite-difference 2D heat step with Neumann BC."""
    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    lap = (
        u_pad[:, 1:-1, 2:] + u_pad[:, 1:-1, :-2] + 
        u_pad[:, 2:, 1:-1] + u_pad[:, :-2, 1:-1] - 4 * u
    )
    return u + alpha * lap


def heat_step_variable(u: torch.Tensor, alpha_map: torch.Tensor) -> torch.Tensor:
    """Heat step with spatially-varying diffusion coefficient."""
    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    lap = (
        u_pad[:, 1:-1, 2:] + u_pad[:, 1:-1, :-2] + 
        u_pad[:, 2:, 1:-1] + u_pad[:, :-2, 1:-1] - 4 * u
    )
    return u + alpha_map.unsqueeze(0) * lap




def generate_sequence(B: int, H: int, W: int, T: int, alpha: float, device) -> torch.Tensor:
    """Generate heat equation sequence with specified alpha."""
    u0 = torch.rand(B, H, W, device=device)
    seq = [u0]
    u = u0
    for _ in range(T - 1):
        u = heat_step(u, alpha)
        seq.append(u)
    return torch.stack(seq, 1)


def generate_variable_sequence(B: int, H: int, W: int, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequence with spatially-varying diffusion."""
    # Create a diffusion map with two regions
    alpha_map = torch.ones(H, W, device=device) * 0.05
    # High diffusion region in center
    h_start, h_end = H // 3, 2 * H // 3
    w_start, w_end = W // 3, 2 * W // 3
    alpha_map[h_start:h_end, w_start:w_end] = 0.15
    
    # Add smooth transition
    for i in range(3):
        alpha_map = F.avg_pool2d(alpha_map.unsqueeze(0).unsqueeze(0), 
                                 kernel_size=3, stride=1, padding=1).squeeze()
    
    u0 = torch.rand(B, H, W, device=device)
    seq = [u0]
    u = u0
    for _ in range(T - 1):
        u = heat_step_variable(u, alpha_map)
        seq.append(u)
    return torch.stack(seq, 1), alpha_map




# ================== Utilities ==================

def gaussian_kernel(k: int, sigma: float, device) -> torch.Tensor:
    """Create gaussian kernel for attention."""
    ax = torch.arange(k, device=device) - (k - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    K = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    K /= K.sum()
    return K


def make_exact_grid_heads(B: int, H: int, W: int, device) -> torch.Tensor:
    """Create heads at exact pixel centers."""
    xs = torch.linspace(0, W-1, W, device=device)
    ys = torch.linspace(0, H-1, H, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    x_norm = (X / (W - 1)) * 2 - 1
    y_norm = (Y / (H - 1)) * 2 - 1
    heads = torch.stack([x_norm, y_norm], dim=-1).reshape(1, H * W, 2)
    return heads.repeat(B, 1, 1)


def norm_to_pixel(coords: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized coords to pixel coords."""
    x = (coords[..., 0] + 1) * 0.5 * (W - 1)
    y = (coords[..., 1] + 1) * 0.5 * (H - 1)
    return y, x


class FourierPE(nn.Module):
    """Fourier positional encoding."""
    def __init__(self, num_freq: int = 8):
        super().__init__()
        self.num_freq = num_freq
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freq).float())

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        xy01 = (xy + 1.0) * 0.5
        angles = xy01.unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        s = torch.sin(angles)
        c = torch.cos(angles)
        out = [xy01, s.reshape(*xy.shape[:-1], -1), c.reshape(*xy.shape[:-1], -1)]
        return torch.cat(out, dim=-1)


# ================== NFTM Models ==================

class NFTMHeatBase(nn.Module):
    """Base NFTM class with common functionality."""
    
    def __init__(self, H: int, W: int, pe_freqs: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.H, self.W = H, W
        self.pe = FourierPE(pe_freqs)
        self.pe_dim = 2 + 2 * 2 * pe_freqs
        self.hidden_dim = hidden_dim
        
    def read_5tap(self, field: torch.Tensor, heads: torch.Tensor):
        """Read center and 4 neighbors."""
        B, _, H, W = field.shape
        device = field.device
        dx = 2.0 / (W - 1)
        dy = 2.0 / (H - 1)
        offsets = torch.tensor([[0,0],[dx,0],[-dx,0],[0,dy],[0,-dy]], device=device).view(1,1,5,2)
        grids = heads.unsqueeze(2) + offsets
        vals = F.grid_sample(field, grids.view(B, -1, 1, 2),
                           align_corners=True, padding_mode='border')
        vals = vals.view(B, 1, heads.shape[1], 5, 1).squeeze(1).squeeze(-1)
        center = vals[..., 0]
        east, west, north, south = vals[..., 1], vals[..., 2], vals[..., 3], vals[..., 4]
        avg4 = (east + west + north + south) / 4.0
        return center, avg4
    
    def write_delta_pixel(self, field: torch.Tensor, heads: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Write delta at pixel centers."""
        B, _, H, W = field.shape
        y, x = norm_to_pixel(heads, H, W)
        iy = torch.clamp(torch.round(y).long(), 0, H-1)
        ix = torch.clamp(torch.round(x).long(), 0, W-1)
        delta_img = torch.zeros(B, H*W, device=field.device)
        idx = (iy * W + ix)
        for b in range(B):
            delta_img[b].scatter_add_(0, idx[b].view(-1), deltas[b].view(-1))
        delta_img = delta_img.view(B, 1, H, W)
        return field + delta_img
    
    def forward(self, f0: torch.Tensor, heads_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """Rollout the model."""
        fields = [f0]
        f = f0
        for heads in heads_seq:
            delta, _ = self.predict_next_at_heads(f, heads)
            f = self.write_delta_pixel(f, heads, delta)
            fields.append(f)
        return fields


class NFTMHeatFixed(NFTMHeatBase):
    """NFTM with fixed alpha (original version)."""
    
    def __init__(self, H: int, W: int, alpha: float = 0.1, **kwargs):
        super().__init__(H, W, **kwargs)
        self.alpha_fixed = alpha
        self.physical_lap = False
        
        in_dim = 1 + self.pe_dim
        self.residual_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    


    def predict_next_at_heads(self, field: torch.Tensor, heads: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        center, avg4 = self.read_5tap(field, heads)
        laplacian = avg4 - center
        if self.physical_lap:
            laplacian = 4.0 * laplacian  # convert to true 5-point Laplacian scale

        # Use fixed alpha
        alpha = torch.as_tensor(self.alpha_fixed, device=field.device, dtype=field.dtype)

        # Physics update
        delta = alpha * laplacian

        # Expand alpha for monitoring
        B, M = laplacian.shape
        alpha_expanded = alpha.expand(B, M)

        dbg = {
            "alpha": alpha_expanded.detach(),
            "laplacian": laplacian,
            "center": center.detach(),
            "avg": avg4.detach(),
            "delta": delta.detach(),
        }
        return delta, dbg

class NFTMHeatLearnable(NFTMHeatBase):
    """NFTM with learnable global alpha"""


    def __init__(self, H: int, W: int, alpha_range: Tuple[float, float] = (0.05, 0.20), physical_lap: bool = False, **kwargs):
        kwargs.pop("physical_lap", None)
        pe_freqs = kwargs.pop("pe_freqs", 8)
        hidden_dim = kwargs.pop("hidden_dim", 128)
        super().__init__(H, W, pe_freqs=pe_freqs, hidden_dim=hidden_dim)

        self.alpha_min, self.alpha_max = alpha_range
        self.physical_lap = physical_lap
        self.learn_spatial_alpha: bool = kwargs.pop("learn_spatial_alpha", False)

        # Learn log(alpha) with wide feasible range; we'll clamp later
        init_alpha = 0.01
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))

        if self.learn_spatial_alpha:
            yy = torch.linspace(-1.0, 1.0, self.H).view(1, 1, self.H, 1)
            xx = torch.linspace(-1.0, 1.0, self.W).view(1, 1, 1, self.W)
            pos = torch.cat([xx.expand(1, 1, self.H, self.W), yy.expand(1, 1, self.H, self.W)], dim=1)
            self.register_buffer('pos_channels', pos)
            self.alpha_net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1)
            )

        self.register_buffer('running_lap_scale', torch.tensor(1.0))
        self.register_buffer('running_delta_scale', torch.tensor(1.0))
        self.momentum = 0.95

    def predict_next_at_heads(self, field: torch.Tensor, heads: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        center, avg4 = self.read_5tap(field, heads)
        laplacian = avg4 - center
        if self.physical_lap:
            laplacian = 4.0 * laplacian

        # Compute alpha (global or spatial) and physics update
        B, M = laplacian.shape
        if getattr(self, 'learn_spatial_alpha', False):
            # Predict per-pixel alpha in [alpha_min, alpha_max]
            pos = self.pos_channels.expand(B, -1, -1, -1)  # B×2×H×W
            alpha_in = torch.cat([field, pos], dim=1)      # B×3×H×W
            a_logits = self.alpha_net(alpha_in)
            a_unit = torch.sigmoid(a_logits)
            alpha_map = self.alpha_min + (self.alpha_max - self.alpha_min) * a_unit
            grid = heads.view(B, -1, 1, 2)
            alpha_used = F.grid_sample(alpha_map, grid, align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)  # B×M
        else:
            alpha_scalar = torch.exp(self.log_alpha)
            alpha_scalar = torch.clamp(alpha_scalar, 1e-3, 1.0)
            alpha_used = alpha_scalar.expand(B, M)

        # Physics update
        delta = alpha_used * laplacian

        # Update running statistics (monitoring only)
        if self.training:
            with torch.no_grad():
                lap_scale = torch.abs(laplacian).mean()
                delta_scale = torch.abs(delta).mean()
                self.running_lap_scale = self.momentum * self.running_lap_scale + (1 - self.momentum) * lap_scale
                self.running_delta_scale = self.momentum * self.running_delta_scale + (1 - self.momentum) * delta_scale

        # Prepare alpha for monitoring (detached) and training (attached)
        alpha_expanded = alpha_used.detach()

        dbg = {
            "alpha": alpha_expanded,
            "alpha_tensor": alpha_used,  # not detached; useful for spatial-alpha loss
            "laplacian": laplacian,
            "center": center.detach(),
            "avg": avg4.detach(),
            "delta": delta.detach(),
        }
        return delta, dbg


# ================== Baseline Models ==================

class ConvNetBaseline(nn.Module):
    """Simple CNN baseline for comparison."""
    
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 5, padding=2)
        )
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u + self.conv(u)
    
    def rollout(self, u0: torch.Tensor, T: int) -> torch.Tensor:
        """Rollout for T steps."""
        seq = [u0]
        u = u0
        for _ in range(T - 1):
            u = self.forward(u)
            seq.append(u)
        return torch.stack(seq, 1)


class UNetBaseline(nn.Module):
    """U-Net baseline for comparison."""

    def __init__(self, channels: List[int] = [1, 16, 32, 64]):
        super().__init__()
        # Encoder conv blocks and pools
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                nn.ReLU(inplace=True),
            ))
            if i < len(channels) - 2:
                self.pools.append(nn.MaxPool2d(2))
        # Decoder upconvs and conv blocks
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(channels) - 2, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i+1], channels[i], 2, stride=2))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(channels[i]*2, channels[i], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i], channels[i], 3, padding=1),
                nn.ReLU(inplace=True),
            ))
        self.final = nn.Conv2d(channels[1], 1, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = u
        skips = []
        # Encoder: conv -> save skip -> pool (except bottom)
        for i, enc in enumerate(self.enc_blocks):
            x = enc(x)
            if i < len(self.enc_blocks) - 1:
                skips.append(x)
                x = self.pools[i](x)
        # Decoder
        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return u + self.final(x)

    def rollout(self, u0: torch.Tensor, T: int) -> torch.Tensor:
        seq = [u0]
        u = u0
        for _ in range(T - 1):
            u = self.forward(u)
            seq.append(u)
        return torch.stack(seq, 1)


# ================== Training Functions ==================

def train_phase_a(model, opt, mse, args, device, alpha_gt=None):
    """Phase A: Teacher-forced training."""
    losses = []
    
    for epoch in range(1, args.epochs_a + 1):
        # Generate data with appropriate alpha
        if alpha_gt is not None:
            gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha_gt, device)
        elif args.mode in ('variable', 'baseline'):
            gt, alpha_map = generate_variable_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
        
        else:
            # Random alpha for learnable mode
            alpha = 0.1 if args.mode == 'learnable_alpha' else 0.1
            gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha, device)
        
        heads_seq = [make_exact_grid_heads(args.batch_size, args.size, args.size, device) 
                    for _ in range(args.timesteps - 1)]
        
        loss = 0.0
        alpha_stats = []
        
        for t in range(args.timesteps - 1):
            field_t = gt[:, t].unsqueeze(1)
            heads = heads_seq[t]
            
            delta_pred, dbg = model.predict_next_at_heads(field_t, heads)
            grid = heads.view(args.batch_size, -1, 1, 2)
            
            gt_next_vals = F.grid_sample(gt[:, t + 1].unsqueeze(1), grid,
                                        align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            gt_curr_vals = F.grid_sample(gt[:, t].unsqueeze(1), grid,
                                        align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            gt_delta_vals = gt_next_vals - gt_curr_vals
            
            # Always use MSE loss
            loss = loss + mse(delta_pred, gt_delta_vals)
            
            # Robust effective alpha logging: median ± IQR over active pixels
            if "laplacian" in dbg and dbg["laplacian"] is not None:
                L = dbg["laplacian"]
                with torch.no_grad():
                    absL = torch.abs(L)
                    thresh = torch.quantile(absL, 0.2)
                    mask = (absL > thresh)
                    eff_alpha = (delta_pred / (L + 1e-8))[mask]
                    if eff_alpha.numel() > 0:
                        med = eff_alpha.median().item()
                        q1 = eff_alpha.quantile(0.25).item()
                        q3 = eff_alpha.quantile(0.75).item()
                        alpha_stats.append((med, q1, q3))
                    else:
                        alpha_stats.append((dbg["alpha"].mean().item(), 0.0, 0.0))
            else:
                alpha_stats.append((dbg["alpha"].mean().item(), 0.0, 0.0))
    
        
        loss = loss / (args.timesteps - 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            
            # Print median±IQR if available
            if alpha_stats and isinstance(alpha_stats[0], tuple):
                meds = [a[0] for a in alpha_stats if isinstance(a, tuple)]
                q1s = [a[1] for a in alpha_stats if isinstance(a, tuple)]
                q3s = [a[2] for a in alpha_stats if isinstance(a, tuple)]
                alpha_med = float(np.mean(meds)) if meds else 0.0
                alpha_q1 = float(np.mean(q1s)) if q1s else 0.0
                alpha_q3 = float(np.mean(q3s)) if q3s else 0.0
                print(f"[Phase A] Epoch {epoch:3d} | Loss {loss.item():.5f} | Alpha med {alpha_med:.4f} (IQR {alpha_q1:.4f}-{alpha_q3:.4f})")
            else:
                alpha_mean = np.mean(alpha_stats) if len(alpha_stats)>0 else 0.0
                print(f"[Phase A] Epoch {epoch:3d} | Loss {loss.item():.5f} | Alpha {alpha_mean:.4f}")
    
    
    return losses


def train_phase_b(model, opt, mse, args, device, alpha_gt=None):
    """Phase B: Rollout training."""
    losses = []
    psnrs = []
    
    for epoch in range(1, args.epochs_b + 1):
        # Generate data
        if alpha_gt is not None:
            gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha_gt, device)
        elif args.mode in ('variable', 'baseline'):
            gt, alpha_map = generate_variable_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
        
        else:
            alpha = 0.1 if args.mode == 'learnable_alpha' else 0.1
            gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha, device)
        
        f0 = gt[:, 0].unsqueeze(1)
        heads_seq = [make_exact_grid_heads(args.batch_size, args.size, args.size, device) 
                    for _ in range(args.timesteps - 1)]
        
        fields = model(f0, heads_seq)
        pred = torch.stack(fields, 1).squeeze(2)
        
        # Point supervision
        loss = 0.0
        for t in range(args.timesteps - 1):
            heads = heads_seq[t]
            grid = heads.view(args.batch_size, -1, 1, 2)
            gt_next_vals = F.grid_sample(gt[:, t + 1].unsqueeze(1), grid,
                                        align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            pred_next_vals = F.grid_sample(pred[:, t + 1].unsqueeze(1), grid,
                                          align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            loss = loss + mse(pred_next_vals, gt_next_vals)
        
        # Additional losses
        mean_loss = ((pred[:, 1:].mean(dim=(2, 3)) - gt[:, 1:].mean(dim=(2, 3))) ** 2).mean()
        frame_loss = ((pred - gt)**2).mean()
        loss = loss / (args.timesteps - 1) + 1e-3 * mean_loss + 5e-4 * frame_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        
        if epoch % 25 == 0:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(((pred - gt) ** 2).mean()).item()
                psnrs.append(psnr)
                print(f"[Phase B] Epoch {epoch:3d} | Loss {loss.item():.5f} | PSNR {psnr:.2f} dB")
    
    return losses, psnrs


def train_baseline(model, opt, mse, args, device):
    """Train baseline models."""
    losses = []
    
    for epoch in range(1, args.epochs_a + args.epochs_b + 1):
        if args.mode == 'baseline':
            gt, alpha_map = generate_variable_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
        else:
            alpha = 0.1
            gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha, device)
        
        u0 = gt[:, 0].unsqueeze(1)
        pred = model.rollout(u0, args.timesteps).squeeze(2)
        
        loss = mse(pred, gt)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(((pred - gt) ** 2).mean()).item()
                print(f"[Baseline] Epoch {epoch:3d} | Loss {loss.item():.5f} | PSNR {psnr:.2f} dB")
    
    return losses

def train_baseline_timeboxed(model, opt, mse, args, device, time_budget_s: float):
    """Time-boxed training loop for ConvNet/UNet baselines."""
    start = time.time()
    steps = 0
    while time.time() - start < time_budget_s:
        if args.mode == 'baseline':
            gt, alpha_map = generate_variable_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
        else:
            alpha = 0.1
            gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha, device)
        u0 = gt[:, 0].unsqueeze(1)
        pred = model.rollout(u0, args.timesteps).squeeze(2)
        loss = mse(pred, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
    return steps

def train_nftm_timeboxed(model, opt, mse, args, device, time_budget_s: float, alpha_gt: Optional[float] = 0.1,
                          split_a: float = 0.3):
    """Time-boxed NFTM training: spend split_a on Phase A, remainder on Phase B."""
    total_start = time.time()
    a_budget = max(0.0, min(1.0, split_a)) * time_budget_s
    b_budget = time_budget_s - a_budget

    def sample_sequence():
        if alpha_gt is not None:
            return generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha_gt, device)
        if args.mode in ('variable', 'baseline'):
            seq, _ = generate_variable_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
            return seq
        
        return generate_sequence(args.batch_size, args.size, args.size, args.timesteps, 0.1, device)

    # Phase A (teacher-forced)
    a_start = time.time()
    while time.time() - a_start < a_budget and time.time() - total_start < time_budget_s:
        gt = sample_sequence()
        heads_seq = [make_exact_grid_heads(args.batch_size, args.size, args.size, device) for _ in range(args.timesteps - 1)]
        loss = 0.0
        for t in range(args.timesteps - 1):
            field_t = gt[:, t].unsqueeze(1)
            heads = heads_seq[t]
            delta_pred, _ = model.predict_next_at_heads(field_t, heads)
            grid = heads.view(args.batch_size, -1, 1, 2)
            gt_next_vals = F.grid_sample(gt[:, t + 1].unsqueeze(1), grid, align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            gt_curr_vals = F.grid_sample(gt[:, t].unsqueeze(1), grid, align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            gt_delta_vals = gt_next_vals - gt_curr_vals
            loss = loss + mse(delta_pred, gt_delta_vals)
        loss = loss / (args.timesteps - 1)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Phase B (rollout)
    b_start = time.time()
    while time.time() - b_start < b_budget and time.time() - total_start < time_budget_s:
        gt = sample_sequence()
        f0 = gt[:, 0].unsqueeze(1)
        heads_seq = [make_exact_grid_heads(args.batch_size, args.size, args.size, device) for _ in range(args.timesteps - 1)]
        fields = model(f0, heads_seq)
        pred = torch.stack(fields, 1).squeeze(2)
        loss = 0.0
        for t in range(args.timesteps - 1):
            heads = heads_seq[t]
            grid = heads.view(args.batch_size, -1, 1, 2)
            gt_next_vals = F.grid_sample(gt[:, t + 1].unsqueeze(1), grid, align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            pred_next_vals = F.grid_sample(pred[:, t + 1].unsqueeze(1), grid, align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
            loss = loss + mse(pred_next_vals, gt_next_vals)
        mean_loss = ((pred[:, 1:].mean(dim=(2, 3)) - gt[:, 1:].mean(dim=(2, 3))) ** 2).mean()
        frame_loss = ((pred - gt) ** 2).mean()
        loss = loss / (args.timesteps - 1) + 1e-3 * mean_loss + 5e-4 * frame_loss
        opt.zero_grad()
        loss.backward()
        opt.step()


# ================== Evaluation Functions =================


def compare_baselines(args, device):
    """Compare NFTM with baseline models."""
    print("\n=== Baseline Comparison ===")
    
    results = {}
    
    # Train and evaluate each model
    all_models = {
        'NFTM-Fixed': NFTMHeatFixed(args.size, args.size),
        'NFTM-Learnable': NFTMHeatLearnable(args.size, args.size, learn_spatial_alpha=True),
        'ConvNet': ConvNetBaseline(),
        'UNet': UNetBaseline()  # always instantiate UNet
    }
    # Optional filtering via CLI
    requested = [m.strip() for m in getattr(args, 'baseline_models', '').split(',') if m.strip()]
    models = all_models if not requested else {k: all_models[k] for k in requested if k in all_models}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Reset RNGs so every baseline sees identical synthetic data
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse = nn.MSELoss()

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Train/evaluate
        start_time = time.time()

        if name == 'NFTM-Fixed':
            # No training needed; evaluate the fixed-physics rollout directly
            with torch.no_grad():
                gt, _ = generate_variable_sequence(1, args.size, args.size, args.timesteps, device)
                f0 = gt[:, 0].unsqueeze(1)
                heads_seq = [make_exact_grid_heads(1, args.size, args.size, device)
                             for _ in range(args.timesteps - 1)]
                fields = model(f0, heads_seq)
                pred = torch.stack(fields, 1).squeeze(2)
                final_psnr = -10.0 * torch.log10(((pred - gt) ** 2).mean()).item()
            train_time = time.time() - start_time
        elif 'NFTM' in name:
            if getattr(args, 'time_budget_s', 0.0) > 0.0:
                train_nftm_timeboxed(model, opt, mse, args, device, args.time_budget_s, alpha_gt=None, split_a=args.time_split_a)
                train_time = time.time() - start_time
                # Evaluate PSNR after time-boxed training
                with torch.no_grad():
                    gt, _ = generate_variable_sequence(1, args.size, args.size, args.timesteps, device)
                    f0 = gt[:, 0].unsqueeze(1)
                    heads_seq = [make_exact_grid_heads(1, args.size, args.size, device) for _ in range(args.timesteps - 1)]
                    fields = model(f0, heads_seq)
                    pred = torch.stack(fields, 1).squeeze(2)
                    final_psnr = -10.0 * torch.log10(((pred - gt) ** 2).mean()).item()
            else:
                # Two-phase training for NFTM (epoch-based)
                train_phase_a(model, opt, mse, args, device, alpha_gt=None)
                _, psnrs = train_phase_b(model, opt, mse, args, device, alpha_gt=None)
                final_psnr = psnrs[-1] if psnrs else 0
                train_time = time.time() - start_time
        else:
            if getattr(args, 'time_budget_s', 0.0) > 0.0:
                train_baseline_timeboxed(model, opt, mse, args, device, args.time_budget_s)
                train_time = time.time() - start_time
            else:
                # Single phase for baselines (epoch-based)
                losses = train_baseline(model, opt, mse, args, device)
                train_time = time.time() - start_time
            # Evaluate final PSNR
            with torch.no_grad():
                gt, _ = generate_variable_sequence(1, args.size, args.size, args.timesteps, device)
                pred = model.rollout(gt[:, 0].unsqueeze(1), args.timesteps).squeeze(2)
                final_psnr = -10.0 * torch.log10(((pred - gt) ** 2).mean()).item()
        
        # Test inference speed
        with torch.no_grad():
            test_input = torch.rand(1, 1, args.size, args.size, device=device)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inf_start = time.time()
            
            if 'NFTM' in name:
                heads = make_exact_grid_heads(1, args.size, args.size, device)
                for _ in range(10):
                    model.predict_next_at_heads(test_input, heads)
            else:
                for _ in range(10):
                    model(test_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inf_time = (time.time() - inf_start) / 10
        
        results[name] = {
            'params': n_params,
            'psnr': final_psnr,
            'train_time': train_time,
            'inf_time': inf_time * 1000  # ms
        }
    # Print comparison table
    print("\n" + "="*70)
    print(f"{'Model':<20} {'Params':<10} {'PSNR (dB)':<12} {'Train (s)':<12} {'Inf (ms)':<10}")
    print("-"*70)
    for name, res in results.items():
        print(f"{name:<20} {res['params']:<10,} {res['psnr']:<12.2f} {res['train_time']:<12.1f} {res['inf_time']:<10.2f}")
    print("="*70)
    
    # Save results
    with open(f'baseline_comparison_{args.size}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def test_learnable_alpha(args, device):
    """Test if model learns correct alpha values using simple MSE supervision over deltas."""
    print("\n=== Learnable Alpha Test (MSE) ===")
    
    # Test different alpha values
    test_alphas = [0.05, 0.10, 0.15, 0.20]
    learned_alphas = {}
    
    # Simple per-alpha training (beta & sigma removed)
    for alpha_true in test_alphas:
        print(f"\nTraining with alpha = {alpha_true}")

        model = NFTMHeatLearnable(args.size, args.size).to(device)
        model.physical_lap = True

        # Adaptive learning rate based on alpha magnitude
        base_lr = 0.001 if alpha_true < 0.1 else 0.002
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if n != "log_alpha"], "lr": base_lr},
            {"params": [model.log_alpha], "lr": base_lr * getattr(args, 'alpha_lr_mult', 1.0)},
        ]
        opt = torch.optim.Adam(param_groups)

        alpha_history = []

        for epoch in range(args.test_alpha_epochs):
            epoch_loss = 0.0
            alpha_preds = []
            for _ in range(args.test_alpha_batches_per_epoch):
                gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha_true, device)
                heads_seq = [make_exact_grid_heads(args.batch_size, args.size, args.size, device)
                             for _ in range(args.timesteps - 1)]

                batch_loss = 0.0
                for t in range(args.timesteps - 1):
                    field_t = gt[:, t].unsqueeze(1)
                    heads = heads_seq[t]
                    delta_pred, dbg = model.predict_next_at_heads(field_t, heads)
                    grid = heads.view(args.batch_size, -1, 1, 2)
                    gt_next_vals = F.grid_sample(gt[:, t + 1].unsqueeze(1), grid,
                                                 align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
                    gt_curr_vals = F.grid_sample(gt[:, t].unsqueeze(1), grid,
                                                 align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
                    gt_delta = gt_next_vals - gt_curr_vals
                    step_loss = F.mse_loss(delta_pred, gt_delta)
                    batch_loss = batch_loss + step_loss
                    alpha_preds.append(dbg["alpha"].mean().item())
                batch_loss = batch_loss / (args.timesteps - 1)
                epoch_loss = epoch_loss + batch_loss

            loss = epoch_loss / max(1, args.test_alpha_batches_per_epoch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            current_alpha = np.mean(alpha_preds) if alpha_preds else float('nan')
            alpha_history.append(current_alpha)

            if epoch > 100 and epoch % 50 == 0:
                for param_group in opt.param_groups:
                    param_group['lr'] *= 0.7

        final_alpha = np.median(alpha_history[-20:])
        error = abs(final_alpha - alpha_true)
        learned_alphas[alpha_true] = final_alpha
        print(f"True α = {alpha_true:.3f}, Learned α = {final_alpha:.3f}, Error = {error:.4f}")
    
    # ------------------------------------------------------------------
    # Optional Curriculum Learning Template (commented out)
    # ------------------------------------------------------------------
    # The idea: start training on shorter temporal horizons (small T) so
    # alpha can be estimated from very local temporal derivatives, then
    # gradually increase sequence length to improve stability over longer
    # rollouts. This was removed from the active path for simplicity, but
    # you can re-enable it by uncommenting and adapting as needed.
    #
    # Example usage plan:
    #   for alpha_true in test_alphas:
    #       model = NFTMHeatLearnable(args.size, args.size).to(device)
    #       model.physical_lap = True
    #       opt = torch.optim.Adam([
    #           {"params": [p for n,p in model.named_parameters() if n != 'log_alpha'], "lr": 0.002},
    #           {"params": [model.log_alpha], "lr": 0.002 * getattr(args,'alpha_lr_mult',1.0)}
    #       ])
    #       time_scales = [3, 5, 7, args.timesteps]
    #       for stage_T in time_scales:
    #           for epoch in range(args.test_alpha_curriculum_epochs):
    #               gt = generate_sequence(args.batch_size, args.size, args.size, stage_T, alpha_true, device)
    #               heads_seq = [make_exact_grid_heads(args.batch_size, args.size, args.size, device)
    #                            for _ in range(stage_T - 1)]
    #               loss = 0.0
    #               for t in range(stage_T - 1):
    #                   field_t = gt[:, t].unsqueeze(1)
    #                   heads = heads_seq[t]
    #                   delta_pred, dbg = model.predict_next_at_heads(field_t, heads)
    #                   grid = heads.view(args.batch_size, -1, 1, 2)
    #                   gt_next_vals = F.grid_sample(gt[:, t+1].unsqueeze(1), grid, align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
    #                   gt_curr_vals = F.grid_sample(gt[:, t].unsqueeze(1), grid, align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
    #                   gt_delta = gt_next_vals - gt_curr_vals
    #                   loss = loss + F.mse_loss(delta_pred, gt_delta)
    #               loss = loss / (stage_T - 1)
    #               opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    #           # (Optional) inspect alpha after each stage:
    #           # est_alpha = torch.exp(model.log_alpha).item(); print(f"Stage T={stage_T}: alpha≈{est_alpha:.4f}")
    #       # After curriculum, evaluate final alpha similarly to main path.
    #
    # Notes / future extensions:
    #   * Could anneal learning rate per stage.
    #   * Could add light L2 prior pushing alpha into [alpha_min, alpha_max].
    #   * For spatial alpha (learn_spatial_alpha=True) replace log_alpha branch
    #     with sampling predicted alpha map at heads and perhaps add TV regularizer.
    # ------------------------------------------------------------------
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    true_vals = list(learned_alphas.keys())
    learned_vals = list(learned_alphas.values())
    errors = [abs(t - l) for t, l in zip(true_vals, learned_vals)]
    
    ax1.scatter(true_vals, learned_vals, s=80, c='tab:blue', label='Learned', zorder=5)
    ax1.plot([0.05, 0.20], [0.05, 0.20], 'r--', label='y=x (Perfect)', alpha=0.7)
    
    for i, (t, l, e) in enumerate(zip(true_vals, learned_vals, errors)):
        ax1.plot([t, t], [t, l], 'gray', alpha=0.3, linewidth=1)
        ax1.annotate(f'{e:.3f}', xy=(t, l), xytext=(t+0.008, l), fontsize=9, ha='left')
    
    ax1.set_xlabel('True α', fontsize=12)
    ax1.set_ylabel('Learned α', fontsize=12)
    ax1.set_title('Learned vs True Diffusion Coefficient', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.04, 0.21)
    ax1.set_ylim(0.04, 0.21)
    
    colors = ['green' if e < 0.01 else 'yellow' if e < 0.02 else 'orange' if e < 0.05 else 'red' for e in errors]
    ax2.bar(range(len(errors)), errors, color=colors)
    ax2.set_xticks(range(len(errors)))
    ax2.set_xticklabels([f'α={t:.2f}' for t in true_vals])
    ax2.set_ylabel('|α_true − α_learned|', fontsize=12)
    ax2.set_title('Absolute Error per Case', fontsize=13)
    ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Excellent (<0.01)')
    ax2.axhline(y=0.02, color='yellow', linestyle='--', alpha=0.5, label='Good (<0.02)')
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='OK (<0.05)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Alpha learning (MSE), MAE={np.mean(errors):.4f}', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig('alpha_learning_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    mean_error = np.mean(errors)
    print(f"\n=== Final Results ===")
    print(f"Mean absolute error: {mean_error:.4f}")
    
    if mean_error < 0.02:
        print("✓ Success! Model learns alpha accurately with plain MSE.")
        print("\nKey insights:")
        print("• Simple MSE on deltas is sufficient here")
        print("• Accurate Laplacian-based target deltas drive parameter recovery")
        print("• Embedding PDE structure reduces need for uncertainty modeling")
    elif mean_error < 0.05:
        print("○ Good results with MSE supervision.")
        print("• Model learns approximate physical parameters")
        print("• Further tuning could improve accuracy")
    else:
        print("△ Learning remains challenging; consider longer Phase A or lr schedule tweaks.")
    
    return learned_alphas


def visualize_results(model, args, device, mode_name=''):
    """Create comprehensive visualization."""
    print("\n=== Generating Visualizations ===")
    
    # Generate test data
    if args.mode == 'variable':
        gt, alpha_map = generate_variable_sequence(1, args.size, args.size, args.timesteps, device)
    else:
        gt = generate_sequence(1, args.size, args.size, args.timesteps, 0.1, device)
    
    f0 = gt[:, 0].unsqueeze(1)
    
    # Run model
    if hasattr(model, 'predict_next_at_heads'):
        heads_seq = [make_exact_grid_heads(1, args.size, args.size, device) 
                    for _ in range(args.timesteps - 1)]
        fields = model(f0, heads_seq)
        pred = torch.stack(fields, 1).squeeze(2)
        
        # Get alpha maps for all columns (use last available for the final frame)
        alpha_maps = []
        for t in range(max(1, args.timesteps - 1)):
            _, dbg = model.predict_next_at_heads(gt[:, t].unsqueeze(1), heads_seq[t])
            alpha_map_t = dbg["alpha"].reshape(1, args.size, args.size)
            alpha_maps.append(alpha_map_t)
    else:
        pred = model.rollout(f0, args.timesteps).squeeze(2)
        alpha_maps = None
    
    # Compute metrics
    errors = torch.abs(pred - gt)
    psnr_per_t = []
    for t in range(args.timesteps):
        mse_t = ((pred[:, t] - gt[:, t]) ** 2).mean()
        psnr_t = -10.0 * torch.log10(mse_t + 1e-10)
        psnr_per_t.append(psnr_t.item())
    
    # Create figure
    add_alpha_row = bool(alpha_maps)
    add_gt_alpha_row = add_alpha_row and (args.mode == 'variable')
    n_rows = 3 + (1 if add_alpha_row else 0) + (1 if add_gt_alpha_row else 0)
    fig, axes = plt.subplots(n_rows, args.timesteps, figsize=(args.timesteps * 1.6, n_rows * 1.6), constrained_layout=True)
    
    if args.timesteps == 1:
        axes = axes.reshape(-1, 1)
    
    vmin, vmax = 0, 1
    
    gt_im = None
    pred_im = None
    err_im = None
    alpha_im = None
    alpha_gt_im = None

    for t in range(args.timesteps):
        # Ground truth
        gt_im = axes[0, t].imshow(gt[0, t].detach().cpu().numpy(), cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, t].set_title(f't={t}', fontsize=8)
        axes[0, t].axis('off')
        
        # Prediction
        pred_im = axes[1, t].imshow(pred[0, t].detach().cpu().numpy(), cmap='hot', vmin=vmin, vmax=vmax)
        axes[1, t].text(0.5, -0.1, f'{psnr_per_t[t]:.1f} dB', transform=axes[1, t].transAxes, ha='center', fontsize=7)
        axes[1, t].axis('off')
        
        # Error
        err_im = axes[2, t].imshow(errors[0, t].detach().cpu().numpy(), cmap='magma', vmin=0, vmax=0.1)
        axes[2, t].axis('off')
        
        # Alpha map (if available)
        row_idx = 3
        if add_alpha_row:
            # Use last available alpha map for the final frame to avoid blanks
            a_idx = min(t, len(alpha_maps) - 1)
            alpha_im = axes[row_idx, t].imshow(alpha_maps[a_idx][0].detach().cpu().numpy(), cmap='viridis', vmin=0.05, vmax=0.20)
            axes[row_idx, t].axis('off')
        if add_gt_alpha_row:
            # Repeat GT alpha for all columns (static over time)
            alpha_gt_im = axes[row_idx + 1, t].imshow(alpha_map.detach().cpu().numpy(), cmap='viridis', vmin=0.05, vmax=0.20)
            axes[row_idx + 1, t].axis('off')
    
    axes[0, 0].set_ylabel('Ground Truth u', fontsize=10)
    axes[1, 0].set_ylabel('Prediction û', fontsize=10)
    axes[2, 0].set_ylabel('|û − u|', fontsize=10)
    if add_alpha_row:
        axes[3, 0].set_ylabel('α(x,y) (pred)', fontsize=10)
    if add_gt_alpha_row:
        axes[4, 0].set_ylabel('α(x,y) (GT)', fontsize=10)

    # Add colorbars using constrained layout
    if gt_im is not None:
        fig.colorbar(gt_im, ax=axes[0, :], fraction=0.046, pad=0.02, label='Value')
    if err_im is not None:
        fig.colorbar(err_im, ax=axes[2, :], fraction=0.046, pad=0.02, label='|Error|')
    if alpha_im is not None:
        fig.colorbar(alpha_im, ax=axes[3, :], fraction=0.046, pad=0.02, label='α (pred)')
    if add_gt_alpha_row and alpha_gt_im is not None:
        fig.colorbar(alpha_gt_im, ax=axes[4, :], fraction=0.046, pad=0.02, label='α (GT)')
    
    plt.suptitle(f'{mode_name}  |  Size {args.size}×{args.size}  |  Mean PSNR {np.mean(psnr_per_t):.2f} dB', fontsize=12)
    plt.savefig(f'visualization_{args.mode}_{args.size}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"PSNR per timestep: {' '.join([f'{p:.1f}' for p in psnr_per_t])}")
    
    return psnr_per_t


# ================== Main Function ==================

def main():
    parser = argparse.ArgumentParser(description='NFTM Heat Equation Experiments')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='learnable_alpha',
               choices=['fixed', 'learnable_alpha', 'variable', 'baseline', 'test_alpha', 'all'],
               help='Experiment mode')
    
    # Model parameters
    parser.add_argument('--size', type=int, default=32, help='Grid size (H=W)')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps')
    parser.add_argument('--eval_timesteps', type=int, default=None, help='Number of timesteps for evaluation/rollout (default: same as training)')
    parser.add_argument('--pe_freqs', type=int, default=8, help='Fourier PE frequencies')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs_a', type=int, default=200, help='Phase A epochs')
    parser.add_argument('--epochs_b', type=int, default=400, help='Phase B epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Evaluation parameters
    # Removed: rollout_steps, train_size, test_size (legacy for deleted modes)
    # Test-alpha tuning knobs
    parser.add_argument('--test_alpha_epochs', type=int, default=400, help='Epochs per beta attempt in test_alpha')
    parser.add_argument('--test_alpha_batches_per_epoch', type=int, default=4, help='Synthetic batches per epoch in test_alpha')
    parser.add_argument('--test_alpha_curriculum_epochs', type=int, default=50, help='Epochs per curriculum stage in test_alpha')
    # Advanced knobs (safe defaults)
    # MSE is the only loss used
    
    # Baseline selection
    parser.add_argument('--baseline_models', type=str, default='NFTM-Fixed,NFTM-Learnable,ConvNet,UNet',
                        help='Comma-separated list of baseline model names to run')
    parser.add_argument('--time_budget_s', type=float, default=0.0,
                        help='Per-model training time budget in seconds for baseline mode (0=use epoch counts)')
    parser.add_argument('--time_split_a', type=float, default=0.3,
                        help='Fraction of time budget for NFTM Phase A when time_budget_s>0')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Size: {args.size}×{args.size}")
    
    # Create output directory
    output_dir = Path(f'results_{args.mode}_{args.size}')
    output_dir.mkdir(exist_ok=True)
    
    # Run experiments based on mode
    if args.mode == 'all':
        # Run all experiments
        print("\n" + "="*50)
        print("Running ALL experiments")
        print("="*50)
        
        # Test learnable alpha
        args_copy = args
        args_copy.mode = 'test_alpha'
        test_learnable_alpha(args_copy, device)
        
        # Baseline comparison
        args_copy.mode = 'baseline'
        compare_baselines(args_copy, device)
        
    # (Stability, super-res, source modes removed.)
        
    elif args.mode == 'test_alpha':
        test_learnable_alpha(args, device)
        
    elif args.mode == 'baseline':
        compare_baselines(args, device)
        
    else:
        # Standard training modes
        if args.mode == 'fixed':
            model = NFTMHeatFixed(args.size, args.size, alpha=0.1).to(device)
            mode_name = "Fixed α=0.1"
        elif args.mode == 'learnable_alpha':
            model = NFTMHeatLearnable(args.size, args.size).to(device)
            mode_name = "Learnable α"
        elif args.mode == 'variable':
            model = NFTMHeatLearnable(args.size, args.size, learn_spatial_alpha=True).to(device)
            mode_name = "Variable α(x,y)"
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        model.physical_lap = True
        
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse = nn.MSELoss()
        
        print(f"\nTraining {mode_name}...")
        print("="*50)
        
        # Phase A
        print("\nPhase A: Teacher-forced training")
        losses_a = train_phase_a(model, opt, mse, args, device)

        # Phase B and visualization: use eval_timesteps if set and mode==variable
        if args.mode == 'variable' and args.eval_timesteps is not None:
            eval_args = argparse.Namespace(**vars(args))
            eval_args.timesteps = args.eval_timesteps
            print(f"\nPhase B: Rollout training (eval horizon: {eval_args.timesteps})")
            losses_b, psnrs = train_phase_b(model, opt, mse, eval_args, device)
            visualize_results(model, eval_args, device, mode_name)
        else:
            print("\nPhase B: Rollout training")
            losses_b, psnrs = train_phase_b(model, opt, mse, args, device)
            visualize_results(model, args, device, mode_name)

        # Save model
        torch.save(model.state_dict(), output_dir / f'model_{args.mode}.pt')
        print(f"\nModel saved to {output_dir}/model_{args.mode}.pt")
    
    print("\n" + "="*50)
    print("Experiments completed!")
    print("="*50)


if __name__ == "__main__":
    main()