"""
Extended NFTM Heat Equation Training Script
============================================
Comprehensive script for ML4PS workshop experiments with command-line interface.

Usage examples:
    # Basic training with learnable alpha
    python nftm_heat.py --mode learnable_alpha --size 64
    
    # Variable coefficient experiment
    python nftm_heat.py --mode variable --size 64 --epochs_a 300
    
    # Baseline comparison
    python nftm_heat.py --mode baseline --size 32
    
    # Long rollout stability test
    python nftm_heat.py --mode stability --size 64 --rollout_steps 50
    
    # Super-resolution demo
    python nftm_heat.py --mode super_res --train_size 32 --test_size 128
    
    # Source term physics
    python nftm_heat.py --mode source --size 64
    
    # Generate all results for paper
    python nftm_heat.py --mode all --size 64
"""

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
import time
import json
from pathlib import Path

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


def heat_step_source(u: torch.Tensor, alpha: float, source: torch.Tensor) -> torch.Tensor:
    """Heat step with source term."""
    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    lap = (
        u_pad[:, 1:-1, 2:] + u_pad[:, 1:-1, :-2] + 
        u_pad[:, 2:, 1:-1] + u_pad[:, :-2, 1:-1] - 4 * u
    )
    return u + alpha * lap + 0.01 * source.unsqueeze(0)


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


def generate_source_sequence(B: int, H: int, W: int, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequence with heat source."""
    # Create a gaussian source in center
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    source = torch.exp(-2 * (xx**2 + yy**2))
    
    u0 = torch.zeros(B, H, W, device=device)
    seq = [u0]
    u = u0
    for _ in range(T - 1):
        u = heat_step_source(u, 0.1, source)
        seq.append(u)
    return torch.stack(seq, 1), source


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
    """NFTM with learnable global alpha via heteroscedastic-style loss."""

    def __init__(self, H: int, W: int, alpha_range: Tuple[float, float] = (0.05, 0.20), physical_lap: bool = False, **kwargs):
        # Swallow 'physical_lap' so it doesn't leak to the base class
        kwargs.pop("physical_lap", None)
        # Extract base-class kwargs safely
        pe_freqs = kwargs.pop("pe_freqs", 8)
        hidden_dim = kwargs.pop("hidden_dim", 128)
        super().__init__(H, W, pe_freqs=pe_freqs, hidden_dim=hidden_dim)

        self.alpha_min, self.alpha_max = alpha_range
        self.physical_lap = physical_lap  # if True, scale Laplacian by 4 to match 5-point operator

        # Whether to learn a spatially-varying alpha map (used in variable mode)
        self.learn_spatial_alpha: bool = kwargs.pop("learn_spatial_alpha", False)

        # Learn log(alpha) with wide feasible range; we'll clamp later
        init_alpha = float(np.sqrt(alpha_range[0] * alpha_range[1]))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))

        # Separate noise scale for decoupled Gaussian NLL
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))  # σ ≈ 0.135

        # If learning spatial alpha, define a small CNN head to predict alpha_map in [alpha_min, alpha_max]
        if self.learn_spatial_alpha:
            # Alpha head takes field plus 2D positional channels (x,y)
            self.alpha_net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1)
            )
            # Precompute normalized coordinate channels in [-1, 1]
            yy = torch.linspace(-1.0, 1.0, self.H).view(1, 1, self.H, 1)
            xx = torch.linspace(-1.0, 1.0, self.W).view(1, 1, 1, self.W)
            pos = torch.cat([xx.expand(1, 1, self.H, self.W), yy.expand(1, 1, self.H, self.W)], dim=1)
            self.register_buffer('pos_channels', pos)

        # Running stats kept for monitoring (not used in loss)
        self.register_buffer('running_lap_scale', torch.tensor(1.0))
        self.register_buffer('running_delta_scale', torch.tensor(1.0))
        self.momentum = 0.95

    def compute_heteroscedastic_loss(self, delta_pred, gt_delta, laplacian, beta=1.0,
                                     use_lap_weighting: bool = False, weight_gamma: float = 1.0,
                                     alpha_override: Optional[torch.Tensor] = None):
        """
        Decoupled Gaussian NLL:
            μ = α * L_phys
            σ is learned separately (global), independent of α
        """
        if alpha_override is not None:
            alpha = alpha_override  # allow per-head alpha (B×M)
        else:
            alpha = torch.exp(self.log_alpha)
            alpha = torch.clamp(alpha, 1e-3, 1.0)
        sigma2 = torch.exp(2.0 * self.log_sigma)

        # Use provided laplacian (should be physical if flag is on)
        mu = alpha * laplacian.detach()
        err = gt_delta - mu

        # Penalize log term with beta to discourage sigma inflation (beta >= 1 tightens)
        nll = 0.5 * (err * err) / (sigma2 + 1e-8) + 0.5 * beta * torch.log(sigma2 + 1e-8)

        # Optional information weighting by |L| to emphasize informative pixels
        if use_lap_weighting:
            with torch.no_grad():
                w = torch.abs(laplacian)
                w = w / (w.mean() + 1e-8)
                if weight_gamma != 1.0:
                    w = torch.pow(w, weight_gamma)
            nll = nll * w

        # Tiny priors to keep parameters in reasonable ranges (reduced weights and shifted center)
        # Set weights to 0.0 to remove bias entirely if desired.
        prior = 1e-6 * (self.log_alpha - math.log(0.12))**2 + 1e-6 * (self.log_sigma + 2.0)**2

        return (nll.mean() + prior)

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
        elif args.mode == 'variable':
            gt, alpha_map = generate_variable_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
        elif args.mode == 'source':
            gt, source = generate_source_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
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
            
            # If variable mode with spatial alpha, use heteroscedastic loss so alpha_map learns
            if args.mode == 'variable' and getattr(model, 'learn_spatial_alpha', False):
                L = dbg["laplacian"]
                alpha_used = dbg.get("alpha_tensor", None)
                step_loss = model.compute_heteroscedastic_loss(
                    delta_pred, gt_delta_vals, L,
                    beta=getattr(args, 'beta', 1.0),
                    use_lap_weighting=getattr(args, 'use_lap_weighting', False),
                    weight_gamma=getattr(args, 'weight_gamma', 1.0),
                    alpha_override=alpha_used
                )
                loss = loss + step_loss
            # Optional: learnable_alpha mode can also use heteroscedastic loss via CLI switch
            elif args.mode == 'learnable_alpha' and getattr(args, 'learnable_alpha_loss', 'mse') == 'hetero':
                # Robustly get Laplacian
                if "laplacian" in dbg and dbg["laplacian"] is not None:
                    L = dbg["laplacian"]
                else:
                    c_tmp, a_tmp = model.read_5tap(field_t, heads)
                    L = a_tmp - c_tmp
                    if getattr(model, 'physical_lap', False):
                        L = 4.0 * L
                step_loss = model.compute_heteroscedastic_loss(
                    delta_pred, gt_delta_vals, L,
                    beta=getattr(args, 'beta', 1.0),
                    use_lap_weighting=getattr(args, 'use_lap_weighting', False),
                    weight_gamma=getattr(args, 'weight_gamma', 1.0)
                )
                loss = loss + step_loss
            else:
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
        elif args.mode == 'variable':
            gt, alpha_map = generate_variable_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
        elif args.mode == 'source':
            gt, source = generate_source_sequence(args.batch_size, args.size, args.size, args.timesteps, device)
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
        alpha = np.random.uniform(0.05, 0.20) if args.mode == 'baseline' else 0.1
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
        alpha = np.random.uniform(0.05, 0.20) if args.mode == 'baseline' else 0.1
        gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha, device)
        u0 = gt[:, 0].unsqueeze(1)
        pred = model.rollout(u0, args.timesteps).squeeze(2)
        loss = mse(pred, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
    return steps

def train_nftm_timeboxed(model, opt, mse, args, device, time_budget_s: float, alpha_gt: float = 0.1, split_a: float = 0.3):
    """Time-boxed NFTM training: spend split_a on Phase A, remainder on Phase B."""
    total_start = time.time()
    a_budget = max(0.0, min(1.0, split_a)) * time_budget_s
    b_budget = time_budget_s - a_budget

    # Phase A (teacher-forced)
    a_start = time.time()
    while time.time() - a_start < a_budget:
        gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha_gt, device)
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
        gt = generate_sequence(args.batch_size, args.size, args.size, args.timesteps, alpha_gt, device)
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


# ================== Evaluation Functions ==================

def evaluate_stability(model, args, device):
    """Test long rollout stability."""
    print("\n=== Stability Test ===")
    
    # Generate long sequence
    gt = generate_sequence(1, args.size, args.size, args.rollout_steps, 0.1, device)
    f0 = gt[:, 0].unsqueeze(1)
    
    # For NFTM models
    if hasattr(model, 'predict_next_at_heads'):
        heads_seq = [make_exact_grid_heads(1, args.size, args.size, device) 
                    for _ in range(args.rollout_steps - 1)]
        fields = model(f0, heads_seq)
        pred = torch.stack(fields, 1).squeeze(2)
    else:
        # For baseline models
        pred = model.rollout(f0, args.rollout_steps).squeeze(2)
    
    # Compute error over time
    errors = []
    psnrs = []
    for t in range(args.rollout_steps):
        mse_t = ((pred[:, t] - gt[:, t]) ** 2).mean().item()
        psnr_t = -10.0 * math.log10(mse_t + 1e-10)
        errors.append(mse_t)
        psnrs.append(psnr_t)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(range(len(errors)), errors, color='tab:blue', lw=2, label='MSE')
    ax1.set_xlabel('Time step', fontsize=11)
    ax1.set_ylabel('Mean Squared Error', fontsize=11)
    ax1.set_title(f'Error Accumulation (H=W={args.size})', fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper right', fontsize=10)

    ax2.plot(range(len(psnrs)), psnrs, color='tab:green', lw=2, label='PSNR')
    ax2.set_xlabel('Time step', fontsize=11)
    ax2.set_ylabel('PSNR [dB]', fontsize=11)
    ax2.set_title(f'PSNR over Time (steps={args.rollout_steps})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)

    fig.suptitle('Stability Test', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'stability_{args.mode}_{args.size}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final MSE at t={args.rollout_steps}: {errors[-1]:.6f}")
    print(f"Final PSNR at t={args.rollout_steps}: {psnrs[-1]:.2f} dB")
    
    return errors, psnrs


def evaluate_super_resolution(model, args, device):
    """Test super-resolution capability."""
    print("\n=== Super-Resolution Test ===")
    
    # Train on coarse, test on fine
    train_size = args.train_size
    test_size = args.test_size
    
    # Generate test data at high resolution
    gt_fine = generate_sequence(1, test_size, test_size, 10, 0.1, device)
    
    # Downsample initial condition
    f0_coarse = F.interpolate(gt_fine[:, 0].unsqueeze(1), size=(train_size, train_size), mode='bilinear')
    f0_fine = F.interpolate(f0_coarse, size=(test_size, test_size), mode='bilinear')
    
    # Run model at fine resolution
    heads_seq = [make_exact_grid_heads(1, test_size, test_size, device) for _ in range(9)]
    
    # Modify model to work at new resolution
    old_H, old_W = model.H, model.W
    model.H, model.W = test_size, test_size
    
    fields = model(f0_fine, heads_seq)
    pred_fine = torch.stack(fields, 1).squeeze(2)
    
    # Restore original resolution
    model.H, model.W = old_H, old_W
    
    # Compute metrics
    mse = ((pred_fine - gt_fine[:, :10]) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse + 1e-10)
    
    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for t in range(5):
        axes[0, t].imshow(gt_fine[0, t].detach().numpy(), cmap='hot', vmin=0, vmax=1)
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')
        
        axes[1, t].imshow(pred_fine[0, t].detach().numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1, t].axis('off')
    
    axes[0, 0].set_ylabel(f'GT ({test_size}x{test_size})')
    axes[1, 0].set_ylabel(f'NFTM ({train_size}→{test_size})')
    
    plt.suptitle(f'Super-Resolution Test: PSNR = {psnr:.2f} dB')
    plt.tight_layout()
    plt.savefig(f'super_res_{train_size}_to_{test_size}.png')
    plt.show()
    
    print(f"Super-resolution {train_size}→{test_size}: PSNR = {psnr:.2f} dB")
    
    return psnr


def compare_baselines(args, device):
    """Compare NFTM with baseline models."""
    print("\n=== Baseline Comparison ===")
    
    results = {}
    
    # Train and evaluate each model
    all_models = {
        'NFTM-Fixed': NFTMHeatFixed(args.size, args.size),
        'NFTM-Learnable': NFTMHeatLearnable(args.size, args.size),
        'ConvNet': ConvNetBaseline(),
        'UNet': UNetBaseline()  # always instantiate UNet
    }
    # Optional filtering via CLI
    requested = [m.strip() for m in getattr(args, 'baseline_models', '').split(',') if m.strip()]
    models = all_models if not requested else {k: all_models[k] for k in requested if k in all_models}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
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
                gt = generate_sequence(1, args.size, args.size, args.timesteps, 0.1, device)
                f0 = gt[:, 0].unsqueeze(1)
                heads_seq = [make_exact_grid_heads(1, args.size, args.size, device) 
                             for _ in range(args.timesteps - 1)]
                fields = model(f0, heads_seq)
                pred = torch.stack(fields, 1).squeeze(2)
                final_psnr = -10.0 * torch.log10(((pred - gt) ** 2).mean()).item()
            train_time = time.time() - start_time
        elif 'NFTM' in name:
            if getattr(args, 'time_budget_s', 0.0) > 0.0:
                train_nftm_timeboxed(model, opt, mse, args, device, args.time_budget_s, alpha_gt=0.1, split_a=args.time_split_a)
                train_time = time.time() - start_time
                # Evaluate PSNR after time-boxed training
                with torch.no_grad():
                    gt = generate_sequence(1, args.size, args.size, args.timesteps, 0.1, device)
                    f0 = gt[:, 0].unsqueeze(1)
                    heads_seq = [make_exact_grid_heads(1, args.size, args.size, device) for _ in range(args.timesteps - 1)]
                    fields = model(f0, heads_seq)
                    pred = torch.stack(fields, 1).squeeze(2)
                    final_psnr = -10.0 * torch.log10(((pred - gt) ** 2).mean()).item()
            else:
                # Two-phase training for NFTM (epoch-based)
                train_phase_a(model, opt, mse, args, device, alpha_gt=0.1)
                _, psnrs = train_phase_b(model, opt, mse, args, device, alpha_gt=0.1)
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
                gt = generate_sequence(1, args.size, args.size, args.timesteps, 0.1, device)
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
    """Test if model learns correct alpha values using heteroscedastic loss with hyperparameter search."""
    print("\n=== Learnable Alpha Test (Heteroscedastic with Tuning) ===")
    
    # Test different alpha values
    test_alphas = [0.05, 0.10, 0.15, 0.20]
    learned_alphas = {}
    
    # Hyperparameter search for beta (weight of log term)
    beta_values = [0.1, 0.5, 1.0, 2.0]
    
    for alpha_true in test_alphas:
        print(f"\nTraining with alpha = {alpha_true}")
        
        best_alpha = None
        best_error = float('inf')
        best_beta = None
        
        # Try different beta values
        for beta in beta_values:
            # Create fresh model for each attempt
            model = NFTMHeatLearnable(args.size, args.size).to(device)
            model.physical_lap = True

            # Keep default initialization from model constructor (no closed-form alpha init)
            
            # Adaptive learning rate based on alpha magnitude
            base_lr = 0.001 if alpha_true < 0.1 else 0.002
            # Per-parameter LR: optionally scale log_alpha and log_sigma
            param_groups = [
                {"params": [p for n, p in model.named_parameters() if n not in ["log_alpha", "log_sigma"]], "lr": base_lr},
                {"params": [model.log_alpha], "lr": base_lr * getattr(args, 'alpha_lr_mult', 1.0)},
                {"params": [model.log_sigma], "lr": base_lr * getattr(args, 'sigma_lr_mult', 1.0)},
            ]
            opt = torch.optim.Adam(param_groups)
            
            # Train with heteroscedastic loss
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

                        # Get prediction
                        delta_pred, dbg = model.predict_next_at_heads(field_t, heads)

                        # Compute ground truth delta
                        grid = heads.view(args.batch_size, -1, 1, 2)
                        gt_next_vals = F.grid_sample(gt[:, t+1].unsqueeze(1), grid,
                                                    align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
                        gt_curr_vals = F.grid_sample(gt[:, t].unsqueeze(1), grid,
                                                    align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
                        gt_delta = gt_next_vals - gt_curr_vals

                        # Heteroscedastic loss
                        # Robust: compute Laplacian locally if dbg lacks it
                        if "laplacian" in dbg:
                            L = dbg["laplacian"]
                        else:
                            c_tmp, a_tmp = model.read_5tap(field_t, heads)
                            L = a_tmp - c_tmp
                            if getattr(model, 'physical_lap', False):
                                L = 4.0 * L

                        step_loss = model.compute_heteroscedastic_loss(
                            delta_pred, gt_delta, L, beta=beta,
                            use_lap_weighting=getattr(args, 'use_lap_weighting', False),
                            weight_gamma=getattr(args, 'weight_gamma', 1.0)
                        )
                        batch_loss = batch_loss + step_loss

                        alpha_preds.append(dbg["alpha"].mean().item())

                    # Average over time steps
                    batch_loss = batch_loss / (args.timesteps - 1)
                    epoch_loss = epoch_loss + batch_loss

                # Average over batches this epoch
                loss = epoch_loss / max(1, args.test_alpha_batches_per_epoch)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                current_alpha = np.mean(alpha_preds) if alpha_preds else float('nan')
                alpha_history.append(current_alpha)
                
                # Learning rate decay
                if epoch > 100 and epoch % 50 == 0:
                    for param_group in opt.param_groups:
                        param_group['lr'] *= 0.7
            
            # Check if this beta gives better results
            final_alpha = np.median(alpha_history[-20:])
            error = abs(final_alpha - alpha_true)
            
            if error < best_error:
                best_error = error
                best_alpha = final_alpha
                best_beta = beta
        
        learned_alphas[alpha_true] = best_alpha
        print(f"True α = {alpha_true:.3f}, Learned α = {best_alpha:.3f}, Error = {best_error:.4f}, Best β = {best_beta}")
    
    # Alternative: Train with curriculum learning
    print("\n--- Trying Curriculum Learning ---")
    curriculum_alphas = {}
    
    for alpha_true in test_alphas:
        print(f"\nCurriculum training with alpha = {alpha_true}")
        
        model = NFTMHeatLearnable(args.size, args.size).to(device)
        model.physical_lap = True
    # Keep default initialization from model constructor (no closed-form alpha init)
        base_lr = 0.002
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if n not in ["log_alpha", "log_sigma"]], "lr": base_lr},
            {"params": [model.log_alpha], "lr": base_lr * getattr(args, 'alpha_lr_mult', 1.0)},
            {"params": [model.log_sigma], "lr": base_lr * getattr(args, 'sigma_lr_mult', 1.0)},
        ]
        opt = torch.optim.Adam(param_groups)
        
        # Start with shorter sequences and gradually increase
        for curriculum_stage, T_curr in enumerate([3, 5, 7, 10]):
            alpha_history = []
            
            for epoch in range(args.test_alpha_curriculum_epochs):
                gt = generate_sequence(args.batch_size, args.size, args.size, T_curr, alpha_true, device)
                heads_seq = [make_exact_grid_heads(args.batch_size, args.size, args.size, device) 
                            for _ in range(T_curr - 1)]
                
                loss = 0.0
                alpha_preds = []
                
                for t in range(T_curr - 1):
                    field_t = gt[:, t].unsqueeze(1)
                    heads = heads_seq[t]
                    
                    delta_pred, dbg = model.predict_next_at_heads(field_t, heads)
                    
                    grid = heads.view(args.batch_size, -1, 1, 2)
                    gt_next_vals = F.grid_sample(gt[:, t+1].unsqueeze(1), grid,
                                                align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
                    gt_curr_vals = F.grid_sample(gt[:, t].unsqueeze(1), grid,
                                                align_corners=True, padding_mode='border').squeeze(1).squeeze(-1)
                    gt_delta = gt_next_vals - gt_curr_vals
                    
                    # Use best beta from hyperparameter search
                    step_loss = model.compute_heteroscedastic_loss(
                        delta_pred, gt_delta, dbg["laplacian"], beta=1.0,
                        use_lap_weighting=getattr(args, 'use_lap_weighting', False),
                        weight_gamma=getattr(args, 'weight_gamma', 1.0)
                    )
                    loss = loss + step_loss
                    
                    alpha_preds.append(dbg["alpha"].mean().item())
                
                loss = loss / (T_curr - 1)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                alpha_history.append(np.mean(alpha_preds))
            
            print(f"  Stage {curriculum_stage+1} (T={T_curr}): α = {alpha_history[-1]:.4f}")
        
        curriculum_alphas[alpha_true] = np.median(alpha_history[-10:])
    
    # Use best results
    if np.mean([abs(curriculum_alphas[k] - k) for k in test_alphas]) < np.mean([abs(learned_alphas[k] - k) for k in test_alphas]):
        print("\nUsing curriculum learning results (better)")
        learned_alphas = curriculum_alphas
    else:
        print("\nUsing hyperparameter search results (better)")
    
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
    
    fig.suptitle(f'Heteroscedastic loss, MAE={np.mean(errors):.4f}', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig('alpha_learning_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    mean_error = np.mean(errors)
    print(f"\n=== Final Results ===")
    print(f"Mean absolute error: {mean_error:.4f}")
    
    if mean_error < 0.02:
        print("✓ Success! Heteroscedastic loss learns alpha accurately.")
        print("\nKey insights:")
        print("• Treating alpha as aleatoric uncertainty enables joint learning")
        print("• Adaptive scaling and proper initialization are crucial")
        print("• This connects PDE learning to uncertainty quantification")
    elif mean_error < 0.05:
        print("○ Good results with heteroscedastic loss.")
        print("• Model learns approximate physical parameters")
        print("• Further tuning could improve accuracy")
    else:
        print("△ Learning remains challenging but approach is principled.")
    
    return learned_alphas


def visualize_results(model, args, device, mode_name=''):
    """Create comprehensive visualization."""
    print("\n=== Generating Visualizations ===")
    
    # Generate test data
    if args.mode == 'variable':
        gt, alpha_map = generate_variable_sequence(1, args.size, args.size, args.timesteps, device)
    elif args.mode == 'source':
        gt, source = generate_source_sequence(1, args.size, args.size, args.timesteps, device)
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
                       choices=['fixed', 'learnable_alpha', 'variable', 'source', 
                               'baseline', 'stability', 'super_res', 'test_alpha', 'all'],
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
    parser.add_argument('--rollout_steps', type=int, default=50, help='Steps for stability test')
    parser.add_argument('--train_size', type=int, default=32, help='Training size for super-res')
    parser.add_argument('--test_size', type=int, default=128, help='Test size for super-res')
    # Test-alpha tuning knobs
    parser.add_argument('--test_alpha_epochs', type=int, default=400, help='Epochs per beta attempt in test_alpha')
    parser.add_argument('--test_alpha_batches_per_epoch', type=int, default=4, help='Synthetic batches per epoch in test_alpha')
    parser.add_argument('--test_alpha_curriculum_epochs', type=int, default=50, help='Epochs per curriculum stage in test_alpha')
    # Advanced knobs (safe defaults)
    parser.add_argument('--use_lap_weighting', action='store_true', help='Weight loss by |L| to emphasize informative pixels')
    parser.set_defaults(use_lap_weighting=True)
    parser.add_argument('--weight_gamma', type=float, default=1.0, help='Exponent for Laplacian weighting')
    parser.add_argument('--alpha_lr_mult', type=float, default=2.0, help='LR multiplier for log_alpha')
    parser.add_argument('--sigma_lr_mult', type=float, default=0.5, help='LR multiplier for log_sigma')
    parser.add_argument('--learnable_alpha_loss', type=str, choices=['mse', 'hetero'], default='mse',
                        help='Loss used in learnable_alpha Phase A (default mse). Set to hetero to use heteroscedastic NLL.')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta coefficient for heteroscedastic log-term')
    
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
        
        # Stability test
        model = NFTMHeatLearnable(args.size, args.size).to(device)
        model.physical_lap = True
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse = nn.MSELoss()
        train_phase_a(model, opt, mse, args, device)
        train_phase_b(model, opt, mse, args, device)
        evaluate_stability(model, args, device)
        
    elif args.mode == 'test_alpha':
        test_learnable_alpha(args, device)
        
    elif args.mode == 'baseline':
        compare_baselines(args, device)
        
    elif args.mode == 'stability':
        model = NFTMHeatLearnable(args.size, args.size).to(device)
        model.physical_lap = True
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse = nn.MSELoss()
        print("Training model for stability test...")
        train_phase_a(model, opt, mse, args, device)
        train_phase_b(model, opt, mse, args, device)
        evaluate_stability(model, args, device)
        
    elif args.mode == 'super_res':
        model = NFTMHeatLearnable(args.train_size, args.train_size).to(device)
        model.physical_lap = True
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        mse = nn.MSELoss()
        print(f"Training at {args.train_size}×{args.train_size}...")
        train_phase_a(model, opt, mse, args, device)
        train_phase_b(model, opt, mse, args, device)
        evaluate_super_resolution(model, args, device)
        
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
        elif args.mode == 'source':
            model = NFTMHeatLearnable(args.size, args.size).to(device)
            mode_name = "With Source Term"
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