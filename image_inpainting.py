#!/usr/bin/env python3
# image_inpainting.py
# NFTM-style iterative inpainting on CIFAR-10 with:
# - MSE data loss
# - random rollouts + curriculum
# - descent guard (backtracking) at eval (optional at train)
# - damping (beta), per-step clip decay, contractive penalty
# - rich logging + saved plots + metrics.json

import os, random, argparse, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision as tv
import torchvision.transforms as T
import matplotlib.pyplot as plt
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency guard
    imageio = None

from metrics import lpips_dist as _metric_lpips
from metrics import psnr as _metric_psnr
from metrics import ssim as _metric_ssim
from metrics import (fid_init, fid_update, fid_compute,
                     kid_init, kid_update, kid_compute)
from unet_model import TinyUNet
import wandb

# -------------------------- Utilities --------------------------

def set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_transform(benchmark, img_size=32):
    if benchmark == "cifar":
        # Normalize to [-1,1] - no resizing needed
        return T.Compose([T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
    else:
        # Resize other datasets to img_size x img_size
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(), 
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

def random_mask(batch, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3):
    """Return mask M (1=known, 0=missing). Mix of random pixels and random square blocks."""
    B, C, H, W = batch.shape
    device = batch.device
    M = torch.ones((B, 1, H, W), device=device)
    # random pixels
    frac = torch.empty(B, 1, 1, 1, device=device).uniform_(*p_missing)
    pix_mask = (torch.rand(B, 1, H, W, device=device) > frac).float()
    M = M * pix_mask
    # random blocks
    for b in range(B):
        if random.random() < block_prob:
            for _ in range(random.randint(min_blocks, max_blocks)):
                sz = random.randint(H//8, H//3)
                y = random.randint(0, H - sz)
                x = random.randint(0, W - sz)
                M[b, :, y:y+sz, x:x+sz] = 0.0
    return M

def corrupt_images(img, M, noise_std=0.3):
    # keep known pixels, corrupt others with noise
    noise = torch.randn_like(img) * noise_std
    return M*img + (1-M)*noise

def clamp_known(I, I_gt, M):
    # enforce known pixels to match ground truth (hard measurement)
    return I*(1-M) + I_gt*M

def psnr(x, y):
    # images in [-1,1]; peak-to-peak = 2
    mse = F.mse_loss(x, y)
    if mse.item() == 0: return torch.tensor(99.0, device=x.device)
    return 10 * torch.log10(4.0 / mse)

def tv_l1(x, weight=1.0):
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return weight*(dx + dy)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def masked_psnr_metric(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    denom = mask_exp.sum().clamp_min(1e-8)
    mse = ((a - b) * mask_exp).pow(2).sum() / denom
    return 10.0 * torch.log10(4.0 / mse)


def masked_metric_mean(metric_fn, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    try:
        vals = metric_fn(a * mask_exp, b * mask_exp, reduction="none")
        frac = mask_exp.mean(dim=(1, 2, 3)).clamp_min(1e-6)
        return (vals / frac).mean()
    except TypeError:
        val = metric_fn(a * mask_exp, b * mask_exp)
        val_tensor = torch.as_tensor(val, device=a.device, dtype=a.dtype)
        frac_scalar = mask.mean().clamp_min(1e-6)
        scale = (1.0 / frac_scalar).to(device=val_tensor.device, dtype=val_tensor.dtype)
        return val_tensor * scale


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def upsample_for_viz(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    x: (C,H,W) or (B,C,H,W)
    Returns x if scale == 1.0, else bilinear-upsampled for readability.
    """
    if scale is None or scale <= 0:
        scale = 1.0
    if abs(scale - 1.0) < 1e-6:
        return x
    if x.dim() == 3:
        x = x.unsqueeze(0)
        out = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        return out.squeeze(0)
    else:
        return F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

# NEW: scale helpers
def downsample_like(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

def upsample_like(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

def downsample_mask_minpool(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Keep 'unknown' (0) dominant when shrinking: min-pool in practice via -maxpool(-mask)."""
    B, C, H, W = mask.shape  # C==1
    if H == size and W == size:
        return mask
    # compute stride/ks for power-of-two downs
    factor = H // size
    if (H % size) == 0 and (W % size) == 0 and factor >= 2:
        m = -F.max_pool2d(-mask, kernel_size=factor, stride=factor)
    else:
        # fallback: bilinear then threshold (keeps hard 0 where any 0 existed)
        m = F.interpolate(mask, size=(size, size), mode='bilinear', align_corners=False)
        m = (m > 0.999).float()
    return m

# --- Pyramid parsing & step allocation ---

def parse_pyramid_arg(pyr: str, final_size: int):
    """Return an increasing list of sizes ending at final_size."""
    if not pyr:
        return [final_size]
    sizes = [int(s) for s in pyr.split(",") if s.strip()]
    sizes = sorted(set([s for s in sizes if 8 <= s <= final_size]))
    if not sizes or sizes[-1] != final_size:
        sizes.append(final_size)
    return sizes

def split_steps_eval(K_total: int, sizes, steps_arg: str | None = None):
    """Steps per level for eval (sum to K_total). If not provided, give 1 to each coarse, rest to finest."""
    if len(sizes) == 1:
        return [K_total]
    if steps_arg:
        steps = [int(s) for s in steps_arg.split(",") if s.strip()]
        assert sum(steps) == K_total and len(steps) == len(sizes), "pyr_steps must match pyramid and sum to K_eval"
        return steps
    L = len(sizes)
    steps = [1]*(L-1) + [K_total - (L-1)]
    steps[-1] = max(steps[-1], 1)
    return steps

def split_steps_train(K_curr: int, sizes, epoch: int):
    """Steps per level for training curriculum; a bit more coarse in very early epochs."""
    if len(sizes) == 1:
        return [K_curr]
    coarse_each = 2 if (epoch <= 2 and K_curr >= 3) else 1
    L = len(sizes)
    total_coarse = min(coarse_each*(L-1), K_curr-1)
    steps = [coarse_each]*(L-1) + [K_curr - total_coarse]
    # Clamp to >=1 and fix any rounding drift
    for i in range(L-1):
        if steps[i] < 1:
            steps[i] = 1
    steps[-1] = max(steps[-1], 1)
    diff = K_curr - sum(steps)
    steps[-1] += diff
    return steps

# -------------------------- Model --------------------------

class TinyController(nn.Module):
    """
    Inputs: concat(I_t (3ch), M (1ch)) -> conv stack
    Outputs:
      - dI: per-pixel correction (3ch), tanh-clamped
      - gate: per-pixel gate in (0,1) (1ch)
      - log_sigma: per-pixel per-channel log-std (3ch) (kept for compatibility)
    """

    def __init__(self, in_ch=4, width=48):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.head_dI   = nn.Conv2d(width, 3, 3, padding=1)
        self.head_gate = nn.Conv2d(width, 1, 3, padding=1)
        self.head_logS = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, I, M):
        x = torch.cat([I, M], dim=1)  # (B,4,H,W)
        h = self.body(x)
        dI   = self.head_dI(h).tanh()
        gate = torch.sigmoid(self.head_gate(h))
        logS = self.head_logS(h)
        return dI, gate, logS


class UNetController(nn.Module):
    def __init__(self, in_ch=4, base=10):
        super().__init__()
        self.unet = TinyUNet(in_ch=in_ch, out_ch=7, base=base)
        # For controller usage we need linear outputs; override the final activation.
        self.unet.activation = nn.Identity()

    def forward(self, I, M):
        x = torch.cat([I, M], dim=1)
        out = self.unet(x)
        dI_raw, gate_raw, logS = torch.split(out, [3, 1, 3], dim=1)
        dI = torch.tanh(dI_raw)
        gate = torch.sigmoid(gate_raw)
        return dI, gate, logS

def nftm_step(I, I_gt, M, controller, beta=0.5, corr_clip=0.2, clip_decay=1.0):
    """One step without guard; returns (I_new, logS)."""
    dI, gate, logS = controller(I, M)
    dI = dI.clamp(-corr_clip*clip_decay, corr_clip*clip_decay)
    I_new = I + beta * gate * dI
    I_new = clamp_known(I_new, I_gt, M)
    return I_new, logS

# -------------------------- Energy & Guard --------------------------

def energy(I, I_gt, tvw=0.01):
    data_term = F.mse_loss(I, I_gt)
    return data_term + tv_l1(I, tvw)

def nftm_step_guarded(I, I_gt, M, controller, beta, corr_clip=0.2, tvw=0.01,
                      max_backtracks=3, shrink=0.5, clip_decay=1.0):
    """Try a step; if energy ↑, shrink beta and retry."""
    with torch.no_grad():
        E0 = energy(I, I_gt, tvw=tvw)
    cur_beta = beta
    for _ in range(max_backtracks+1):
        I_prop, logS = nftm_step(I, I_gt, M, controller, beta=cur_beta,
                                 corr_clip=corr_clip, clip_decay=clip_decay)
        with torch.no_grad():
            E1 = energy(I_prop, I_gt, tvw=tvw)
        if E1 <= E0:
            return I_prop, logS, cur_beta, True, float(E1 - E0)
        cur_beta *= shrink
    # give up: return original (reject)
    return I, torch.zeros_like(I), cur_beta, False, 0.0

# -------------------------- Train / Eval --------------------------

def train_epoch(controller, opt, loader, device, epoch, K_target=10, K_base=4,
                beta=0.4, beta_max=0.6, tvw=0.01, p_missing=(0.25,0.5), block_prob=0.5, noise_std=0.3,
                corr_clip=0.2, guard_in_train=True, contract_w=1e-3, rollout_bias=True, pyramid_sizes=None):
    controller.train()
    psnrs, losses = [], []
    accepted_steps, backtracks = 0, 0

    # curriculum on rollout depth: grow K_train with epochs
    K_train = min(K_target, K_base + epoch)
    sizes_default = pyramid_sizes if pyramid_sizes else None

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)  # ground truth in [-1,1]
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std)
        I = clamp_known(I0.clone(), imgs, M)

        # random rollout depth (bias early epochs to short rollouts)
        if rollout_bias:
            lengths = list(range(1, K_train+1))
            weights = torch.tensor([0.6*(0.7**(t-1)) for t in lengths])
            weights = (weights / weights.sum()).to(device='cpu')
            K_curr = int(torch.multinomial(weights, 1).item() + 1)  # 1..K_train
        else:
            K_curr = random.randint(1, K_train)

        sizes = sizes_default or [imgs.shape[-1]]
        steps_per = split_steps_train(K_curr, sizes, epoch=epoch)

        I = clamp_known(I0.clone(), imgs, M)
        I_prev_for_contract = I.clone().detach()

        for lvl, (S, T) in enumerate(zip(sizes, steps_per)):
            gt_S = imgs if S == imgs.shape[-1] else downsample_like(imgs, S)
            M_S = M if S == imgs.shape[-1] else downsample_mask_minpool(M, S)
            I = I if I.shape[-1] == S else downsample_like(I, S)

            scale_fac = float(sizes[-1]) / float(S)
            beta_S = min(beta * scale_fac, beta_max)
            corr_clip_S = corr_clip * scale_fac

            for s in range(T):
                clip_decay = (0.92 ** s)
                if guard_in_train:
                    I, _, _used_beta, ok, _dE = nftm_step_guarded(
                        I, gt_S, M_S, controller,
                        beta=beta_S, corr_clip=corr_clip_S,
                        tvw=0.0, max_backtracks=2, shrink=0.5, clip_decay=clip_decay
                    )
                    accepted_steps += int(ok)
                    backtracks += int(not ok)
                else:
                    I, _ = nftm_step(
                        I, gt_S, M_S, controller,
                        beta=beta_S, corr_clip=corr_clip_S, clip_decay=clip_decay
                    )

            if S != sizes[-1]:
                nextS = sizes[lvl+1]
                I = upsample_like(I, nextS)
                gt_next = imgs if nextS == imgs.shape[-1] else downsample_like(imgs, nextS)
                M_next  = M if nextS == M.shape[-1] else downsample_mask_minpool(M, nextS)
                I = clamp_known(I, gt_next, M_next)

        # Loss on final fine output
        data_loss = F.mse_loss(I, imgs)

        # Regularizers
        loss_smooth = tv_l1(I, tvw)
        contract = contract_w * (I - I_prev_for_contract).pow(2).mean()

        loss = data_loss + loss_smooth + contract

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            losses.append(loss.item())
            psnrs.append(psnr(I, imgs).item())

    stats = dict(train_K=K_train, accepted=accepted_steps, backtracks=backtracks)
    return float(np.mean(losses)), float(np.mean(psnrs)), stats

@torch.no_grad()
def eval_steps(controller, loader, device, K_eval=10, beta=0.6,
               p_missing=(0.25,0.5), block_prob=0.5, noise_std=0.3,
               corr_clip=0.2, descent_guard=False, tvw=0.0,
               save_per_epoch_dir=None, epoch_tag=None, pyramid_sizes=None, 
               steps_split=None, viz_scale: float = 1.0):
    controller.eval()
    psnrs_step, ssims_step, lpips_step = [], [], []
    # optional per-epoch visualization of first batch progression
    save_seq = (save_per_epoch_dir is not None)
    if save_seq:
        ensure_dir(save_per_epoch_dir)

    for bidx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std)
        I = clamp_known(I0.clone(), imgs, M)
        step_psnrs, step_ssims, step_lpips = [], [], []

        gif_frames = []
        make_gif_frame = None

        if save_seq and bidx == 0:
            vis_rows = min(6, imgs.size(0))
            cols = K_eval + 2
            plt.figure(figsize=(3*cols, 3*vis_rows))

            def show_img(ax, x_tensor):
                vis = upsample_for_viz(x_tensor, viz_scale)
                ax.imshow(((vis.permute(1, 2, 0).cpu().numpy()+1)/2).clip(0,1))
                ax.axis('off')

            gif_cols = min(8, imgs.size(0))
            gif_cols = max(gif_cols, 1)
            gif_gt = imgs[:gif_cols].detach()

            def make_gif_frame(batch):
                if imageio is None:
                    return None
                current = batch[:gif_cols].detach().clamp(-1.0, 1.0)
                gt = gif_gt.clamp(-1.0, 1.0)
                up_gt = upsample_for_viz(gt, viz_scale)
                up_cur = upsample_for_viz(current, viz_scale)
                panel = torch.cat([up_gt, up_cur], dim=0)
                panel = ((panel + 1.0) * 0.5).clamp(0.0, 1.0)
                grid = tv.utils.make_grid(panel, nrow=gif_cols, padding=2, pad_value=0.0)
                return (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            for r in range(vis_rows):
                ax = plt.subplot(vis_rows, cols, r*cols+1)
                show_img(ax, imgs[r])
                if r == 0:
                    ax.set_title("GT")
                ax = plt.subplot(vis_rows, cols, r*cols+2)
                show_img(ax, I[r])
                if r == 0:
                    ax.set_title("Init")

            if imageio is not None:
                init_frame = make_gif_frame(I.clamp(-1.0, 1.0))
                if init_frame is not None:
                    gif_frames.append(init_frame)

        # BEFORE (conceptually):
        # for s in range(K_eval): step at native resolution

        # --- Multi-scale rollout (fills per-step curves) ---
        sizes = pyramid_sizes or [imgs.shape[-1]]
        steps_per = steps_split or split_steps_eval(K_eval, sizes)

        I = clamp_known(I0.clone(), imgs, M)
        total_steps = 0  # to ensure we produce exactly K_eval entries

        for lvl, (S, T) in enumerate(zip(sizes, steps_per)):
            gt_S = imgs if S == imgs.shape[-1] else downsample_like(imgs, S)
            M_S = M if S == imgs.shape[-1] else downsample_mask_minpool(M, S)
            I = I if I.shape[-1] == S else downsample_like(I, S)

            scale_fac = float(sizes[-1]) / float(S)
            beta_S = min(beta * scale_fac, 0.9)
            corr_clip_S = corr_clip * scale_fac

            for s in range(T):
                clip_decay = (0.92 ** s)
                if descent_guard:
                    I, _, _, _, _ = nftm_step_guarded(
                        I, gt_S, M_S, controller, beta=beta_S, corr_clip=corr_clip_S,
                        tvw=tvw, max_backtracks=3, shrink=0.5, clip_decay=clip_decay
                    )
                else:
                    I, _ = nftm_step(I, gt_S, M_S, controller, beta=beta_S,
                                     corr_clip=corr_clip_S, clip_decay=clip_decay)
                    
                if save_seq and bidx == 0:
                    vis_rows = min(6, imgs.size(0))
                    cols = K_eval + 2
                    show_tensor = I if I.shape[-1] == imgs.shape[-1] else upsample_like(I, imgs.shape[-1])
                    for r in range(vis_rows):
                        ax = plt.subplot(vis_rows, cols, r*cols + (total_steps + 3))
                        show_img(ax, show_tensor[r])
                        if r == 0:
                            ax.set_title(f"step {total_steps+1}")

                # --- per-step metrics at native size ---
                I_metrics = I if I.shape[-1] == imgs.shape[-1] else upsample_like(I, imgs.shape[-1])
                I_metrics = I_metrics.clamp(-1.0, 1.0)
                step_psnrs.append(_metric_psnr(I_metrics, imgs).item())
                step_ssims.append(_metric_ssim(I_metrics, imgs).item())
                step_lpips.append(_metric_lpips(I_metrics, imgs).item())
                total_steps += 1

                if save_seq and bidx == 0 and imageio is not None and make_gif_frame is not None:
                    frame = make_gif_frame(I_metrics)
                    if frame is not None:
                        gif_frames.append(frame)

            if S != sizes[-1]:
                nextS = sizes[lvl+1]
                I = upsample_like(I, nextS)
                gt_next = imgs if nextS == imgs.shape[-1] else downsample_like(imgs, nextS)
                M_next  = M if nextS == M.shape[-1] else downsample_mask_minpool(M, nextS)
                I = clamp_known(I, gt_next, M_next)


        psnrs_step.append(step_psnrs)
        ssims_step.append(step_ssims)
        lpips_step.append(step_lpips)

        if save_seq and bidx == 0:
            plt.tight_layout()
            tag = f"epoch_{epoch_tag}" if epoch_tag is not None else "eval"
            out_path = os.path.join(save_per_epoch_dir, f"progress_{tag}.png")
            plt.savefig(out_path, dpi=140)
            print(f"[viz] saved per-epoch progression → {out_path}")
            plt.close()
            if imageio is not None and gif_frames:
                gif_path = os.path.join(save_per_epoch_dir, f"progress_{tag}.gif")
                imageio.mimsave(gif_path, gif_frames, duration=0.4)
                print(f"[gif] saved reconstruction GIF (GT top row, recon bottom) → {gif_path}")
            elif imageio is None:
                print("[gif] skipped GIF generation (imageio not installed)")

    curves = {
        "psnr": np.array(psnrs_step).mean(axis=0) if psnrs_step else np.array([]),
        "ssim": np.array(ssims_step).mean(axis=0) if ssims_step else np.array([]),
        "lpips": np.array(lpips_step).mean(axis=0) if lpips_step else np.array([]),
    }
    return curves


@torch.no_grad()
def evaluate_metrics_full(
    controller,
    loader,
    device,
    *,
    K_eval: int,
    beta: float,
    p_missing=(0.25, 0.5),
    block_prob=0.5,
    noise_std=0.3,
    corr_clip=0.2,
    descent_guard: bool = False,
    tvw: float = 0.0,
    benchmark=None,
    pyramid_sizes=None,
    steps_split=None,
):
    controller.eval()
    totals = {
        "psnr_all": 0.0,
        "psnr_miss": 0.0,
        "ssim_all": 0.0,
        "ssim_miss": 0.0,
        "lpips_all": 0.0,
        "lpips_miss": 0.0,
    }
    batches = 0
    fid_metric = fid_init(device)
    kid_metric = kid_init(device, benchmark=benchmark)

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        M = random_mask(imgs, p_missing=p_missing, block_prob=block_prob).to(device)
        I0 = corrupt_images(imgs, M, noise_std=noise_std)

        # multi-scale rollout, same as eval_steps()
        sizes = pyramid_sizes or [imgs.shape[-1]]
        steps_per = steps_split or split_steps_eval(K_eval, sizes)

        I = clamp_known(I0.clone(), imgs, M)

        for lvl, (S, T) in enumerate(zip(sizes, steps_per)):
            gt_S = imgs if S == imgs.shape[-1] else downsample_like(imgs, S)
            M_S  = M if S == imgs.shape[-1] else downsample_mask_minpool(M, S)
            I    = I if I.shape[-1] == S else downsample_like(I, S)

            scale_fac = float(sizes[-1]) / float(S)
            beta_S = min(beta * scale_fac, 0.9)
            corr_clip_S = corr_clip * scale_fac

            for s in range(T):
                clip_decay = 0.92 ** s
                if descent_guard:
                    I, _, _, _, _ = nftm_step_guarded(I, gt_S, M_S, controller,
                                                      beta=beta_S, corr_clip=corr_clip_S,
                                                      tvw=tvw, max_backtracks=3, shrink=0.5,
                                                      clip_decay=clip_decay)
                else:
                    I, _ = nftm_step(I, gt_S, M_S, controller, beta=beta_S,
                                     corr_clip=corr_clip_S, clip_decay=clip_decay)

            # upsample hand-off with size-matched clamp
            if S != sizes[-1]:
                nextS = sizes[lvl+1]
                I = upsample_like(I, nextS)
                gt_next = imgs if nextS == imgs.shape[-1] else downsample_like(imgs, nextS)
                M_next  = M if nextS == imgs.shape[-1] else downsample_mask_minpool(M, nextS)
                I = clamp_known(I, gt_next, M_next)

        preds = I.clamp(-1.0, 1.0)
        miss_mask = 1.0 - M

        totals["psnr_all"] += float(_metric_psnr(preds, imgs).item())
        totals["psnr_miss"] += float(masked_psnr_metric(preds, imgs, miss_mask).item())
        totals["ssim_all"] += float(_metric_ssim(preds, imgs).item())
        totals["ssim_miss"] += float(masked_metric_mean(_metric_ssim, preds, imgs, miss_mask).item())
        totals["lpips_all"] += float(_metric_lpips(preds, imgs).item())
        totals["lpips_miss"] += float(masked_metric_mean(_metric_lpips, preds, imgs, miss_mask).item())
        batches += 1
        # Accumulate FID/KID features
        fid_update(fid_metric, imgs, preds)
        kid_update(kid_metric, imgs, preds)


    if batches == 0:
        raise RuntimeError("Evaluation loader produced no batches for metric computation.")

    results = {k: totals[k] / batches for k in totals}
    # Compute final FID/KID scores over full test set
    results["fid"] = fid_compute(fid_metric)
    results["kid"] = kid_compute(kid_metric)

    return results


def plot_metric_curve(curve, save_path, ylabel, title):
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(curve)+1), curve, marker='o')
    plt.xlabel("NFTM step"); plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=160)
    print(f"[plot] saved {title.lower()} → {save_path}")
    plt.close()

# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(description="NFTM-style iterative inpainting on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--K_train", type=int, default=8, help="max rollout steps for training curriculum")
    parser.add_argument("--K_eval", type=int, default=12, help="rollout steps for evaluation")
    parser.add_argument("--beta_start", type=float, default=0.28, help="initial beta (step size)")
    parser.add_argument("--beta_max", type=float, default=0.6, help="cap on beta during training")
    parser.add_argument("--beta_anneal", type=float, default=0.03, help="per-epoch beta increment")
    parser.add_argument("--beta_eval_bonus", type=float, default=0.05, help="extra beta for eval")
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--corr_clip", type=float, default=0.1, help="max per-step correction magnitude (base)")
    parser.add_argument("--pmin", type=float, default=0.25, help="min missing fraction")
    parser.add_argument("--pmax", type=float, default=0.5, help="max missing fraction")
    parser.add_argument("--block_prob", type=float, default=0.5, help="probability to add random occlusion blocks")
    parser.add_argument("--noise_std", type=float, default=0.3, help="corruption noise std for missing pixels")
    parser.add_argument("--width", type=int, default=48, help="controller width")
    parser.add_argument("--controller", type=str, default="dense", choices=["dense", "unet"],
                        help="controller architecture")
    parser.add_argument("--unet_base", type=int, default=10, help="base channels for UNet controller")
    parser.add_argument("--save_dir", type=str, default="out", help="directory to save plots/metrics")
    parser.add_argument("--save_epoch_progress", action="store_true", help="save per-epoch step grids for the first eval batch")
    parser.add_argument("--guard_in_train", action="store_true", help="enable descent guard during training (slower, more stable)")
    parser.add_argument("--contract_w", type=float, default=1e-3, help="contractive penalty weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", type=str, default="cifar", choices=["cifar", "set12", "cbsd68", "celebahq"],help="choose test dataset for benchmarking")
    parser.add_argument("--train_dataset", type=str, default="cifar", choices=["cifar", "celebahq"], help="Dataset for training")
    parser.add_argument("--img_size", type=int, default=32, choices=[32, 64], help="Input image size (resize if necessary)")

    parser.add_argument("--save_metrics", action="store_true", help="save metrics.json + psnr_curve.npy to save_dir")
    parser.add_argument("--use_wandb", action="store_true", help="enable logging to Weights & Biases (wandb)")
    parser.add_argument("--pyramid", type=str, default="", help="comma-separated sizes for coarse->fine (e.g., '16,32' or '16,32,64'). Empty = single-scale.")
    parser.add_argument("--pyr_steps", type=str, default="", help="comma-separated rollout steps per level summing to K_eval (e.g., '3,9'). Empty = auto split.")
    parser.add_argument("--viz_scale", type=float, default=1.0, help="Visualization upsample scale for PNG/GIF (1.0 = native, 2.0 = 2×).")


    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    log_with_wandb = args.use_wandb
    print(f"[device] {device} | criterion=MSE")

    # Data
    train_dataset_name = args.train_dataset.lower()
    img_size = args.img_size
    benchmark = args.benchmark.lower()

    # Parse pyramid config once
    pyr_sizes = parse_pyramid_arg(args.pyramid, img_size)
    pyr_steps_eval = split_steps_eval(args.K_eval, pyr_sizes, args.pyr_steps if args.pyr_steps else None)
    
    # Set training dataset
    if train_dataset_name == "cifar":
        transform_train = get_transform("cifar", img_size=img_size)
        train_set = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    elif train_dataset_name == "celebahq":
        transform_train = get_transform("celebahq", img_size=img_size)
        train_set = ImageFolder(root="./benchmarks/CelebAHQ/", transform=transform_train)
    else:
        raise ValueError(f"Unknown train dataset: {args.train_dataset}")
    
    transform_test = get_transform(benchmark, img_size=img_size)
    # Set test dataset
    if benchmark == "cifar":
        test_set = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif benchmark == "set12":
        test_set = ImageFolder(root="./benchmarks/Set12", transform=transform_test)
    elif benchmark == "cbsd68":
        test_set = ImageFolder(root="./benchmarks/CBSD68", transform=transform_test)
    elif (benchmark == "celebahq") and train_dataset_name == "celebahq":
        # Random split 80-20 for train and test set
        train_size = int(0.8 * len(train_set))
        test_size = len(train_set) - train_size
        train_set, test_set = random_split(train_set, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    elif benchmark == "celebahq":
        test_set = ImageFolder(root="./benchmarks/CelebAHQ", transform=transform_test)
    else:
        raise ValueError(f"Unknown benchmark dataset: {args.benchmark}")

    use_cuda_pinning = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2,
                              pin_memory=use_cuda_pinning)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              pin_memory=use_cuda_pinning)

    # Model / Optim
    controller_info = {"name": args.controller}
    if args.controller == "dense":
        controller = TinyController(in_ch=4, width=args.width).to(device)
    else:
        target = count_params(TinyController(in_ch=4, width=args.width))
        base = args.unet_base
        best = None
        for b in range(6, 14):
            tmp = UNetController(in_ch=4, base=b)
            n = count_params(tmp)
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (b, n)
        if best is not None and abs(best[1] - target) <= 0.05 * target and best[0] != base:
            print(f"[controller] auto-adjust unet_base {base} -> {best[0]} (target params={target})")
            base = best[0]
        controller = UNetController(in_ch=4, base=base).to(device)
        controller_info["base"] = base
    param_total = count_params(controller)
    controller_info["params"] = param_total
    info_msg = f"[controller] {controller_info['name']}"
    if "base" in controller_info:
        info_msg += f" | base={controller_info['base']}"
    info_msg += f" | params={controller_info['params']}"
    print(info_msg)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ensure_dir(args.save_dir)
    steps_dir = os.path.join(args.save_dir, "steps") if args.save_epoch_progress else None
    if steps_dir: ensure_dir(steps_dir)

    psnr_curve = None
    ssim_curve = None
    lpips_curve = None

    # Train
    for ep in range(1, args.epochs+1):
        beta = min(args.beta_start + args.beta_anneal * (ep-1), args.beta_max)
        beta_eval = min(beta + args.beta_eval_bonus, 0.9)

        train_loss, train_psnr, stats = train_epoch(
            controller, opt, train_loader, device, epoch=ep,
            K_target=args.K_train, K_base=4, beta=beta, beta_max=args.beta_max, tvw=args.tv_weight,
            p_missing=(args.pmin, args.pmax), block_prob=args.block_prob,
            noise_std=args.noise_std, corr_clip=args.corr_clip,
            guard_in_train=args.guard_in_train,
            contract_w=args.contract_w, rollout_bias=True,
            pyramid_sizes=pyr_sizes
        )

        curves = eval_steps(
            controller, test_loader, device,
            K_eval=args.K_eval, beta=beta_eval,
            p_missing=(args.pmin, args.pmax), block_prob=args.block_prob,
            noise_std=args.noise_std, corr_clip=args.corr_clip,
            descent_guard=False, tvw=0.0,
            save_per_epoch_dir=steps_dir, epoch_tag=ep,
            pyramid_sizes=pyr_sizes, steps_split=pyr_steps_eval,
            viz_scale=max(1.0, float(args.viz_scale))
        )

        psnr_curve = curves["psnr"]
        ssim_curve = curves["ssim"]
        lpips_curve = curves["lpips"]

        if psnr_curve.size > 0:
            head = psnr_curve[:min(5, len(psnr_curve))]
            curve_str = ", ".join(f"{v:.2f}" for v in head)
            tail_val = f"{psnr_curve[-1]:.2f}"
        else:
            curve_str = "n/a"
            tail_val = "n/a"
        msg = (f"[ep {ep:02d}] β_train={beta:.3f} K_train={stats['train_K']} | loss {train_loss:.4f} | "
               f"train PSNR {train_psnr:.2f} dB | eval PSNR 1..{args.K_eval}: {curve_str} ... {tail_val} | "
               f"ctrl={args.controller}")
        if ssim_curve.size > 0 and lpips_curve.size > 0:
            msg += (f" | final SSIM {ssim_curve[-1]:.4f} | final LPIPS {lpips_curve[-1]:.4f}")
        print(msg)
        if args.guard_in_train:
            print(f"         accepted steps: {stats['accepted']} | backtracks (approx): {stats['backtracks']}")
        # Log metrics to Weights & Biases
        if log_with_wandb:
            wandb.log({
                "epoch": ep,
                "train/loss": train_loss,
                "train/psnr": train_psnr,
                "eval/psnr_final": float(psnr_curve[-1]) if psnr_curve.size > 0 else None,
                "eval/ssim_final": float(ssim_curve[-1]) if ssim_curve.size > 0 else None,
                "eval/lpips_final": float(lpips_curve[-1]) if lpips_curve.size > 0 else None,
                "beta": beta,
                "K_train": stats["train_K"],
                "controller": args.controller,
            })

    # Save final metric curves
    if psnr_curve is not None and psnr_curve.size > 0:
        plot_metric_curve(psnr_curve, os.path.join(args.save_dir, "psnr_curve.png"),
                          "PSNR (dB)", "Step-wise PSNR (eval)")
    if ssim_curve is not None and ssim_curve.size > 0:
        plot_metric_curve(ssim_curve, os.path.join(args.save_dir, "ssim_curve.png"),
                          "SSIM", "Step-wise SSIM (eval)")
    if lpips_curve is not None and lpips_curve.size > 0:
        plot_metric_curve(lpips_curve, os.path.join(args.save_dir, "lpips_curve.png"),
                          "LPIPS", "Step-wise LPIPS (eval)")

    # Save a final sample grid (progression of first test batch)
    final_beta = min(args.beta_start + args.beta_anneal * (args.epochs-1), args.beta_max)
    beta_eval_final = min(final_beta + args.beta_eval_bonus, 0.9)

    _ = eval_steps(
        controller, test_loader, device,
        K_eval=args.K_eval, beta=beta_eval_final,
        p_missing=(args.pmin, args.pmax), block_prob=args.block_prob,
        noise_std=args.noise_std, corr_clip=args.corr_clip,
        descent_guard=False, tvw=0.0,
        save_per_epoch_dir=os.path.join(args.save_dir, "final"),
        epoch_tag="final",
        pyramid_sizes=pyr_sizes, steps_split=pyr_steps_eval,
        viz_scale=max(1.0, float(args.viz_scale))
    )
    print("[done] checkpoints and plots saved under:", args.save_dir, f"| controller={args.controller}")

    # Save metrics for the driver / comparisons
    if args.save_metrics:
        metrics_full = evaluate_metrics_full(
            controller,
            test_loader,
            device,
            K_eval=args.K_eval,
            beta=beta_eval_final,
            p_missing=(args.pmin, args.pmax),
            block_prob=args.block_prob,
            noise_std=args.noise_std,
            corr_clip=args.corr_clip,
            descent_guard=False,
            tvw=0.0,
            benchmark=benchmark,
            pyramid_sizes=pyr_sizes,
            steps_split=pyr_steps_eval,
        )
        if psnr_curve is not None and psnr_curve.size > 0:
            np.save(os.path.join(args.save_dir, "psnr_curve.npy"), psnr_curve)
        if ssim_curve is not None and ssim_curve.size > 0:
            np.save(os.path.join(args.save_dir, "ssim_curve.npy"), ssim_curve)
        if lpips_curve is not None and lpips_curve.size > 0:
            np.save(os.path.join(args.save_dir, "lpips_curve.npy"), lpips_curve)
        summary = dict(
            epochs=args.epochs,
            K_train=args.K_train,
            K_eval=args.K_eval,
            beta_start=args.beta_start,
            beta_max=args.beta_max,
            beta_anneal=args.beta_anneal,
            beta_eval_bonus=args.beta_eval_bonus,
            corr_clip=args.corr_clip,
            tv_weight=args.tv_weight,
            pmin=args.pmin, pmax=args.pmax, block_prob=args.block_prob,
            noise_std=args.noise_std,
            final_psnr=float(psnr_curve[-1]) if psnr_curve is not None and psnr_curve.size > 0 else None,
        )
        if ssim_curve is not None and ssim_curve.size > 0:
            summary["final_ssim"] = float(ssim_curve[-1])
        if lpips_curve is not None and lpips_curve.size > 0:
            summary["final_lpips"] = float(lpips_curve[-1])
        summary["curve"] = [float(x) for x in psnr_curve] if psnr_curve is not None and psnr_curve.size > 0 else []
        if ssim_curve is not None and ssim_curve.size > 0:
            summary["curve_ssim"] = [float(x) for x in ssim_curve]
        if lpips_curve is not None and lpips_curve.size > 0:
            summary["curve_lpips"] = [float(x) for x in lpips_curve]
        summary.update(metrics_full)
        summary["params"] = float(param_total)
        summary["seed"] = int(args.seed)
        summary["controller"] = args.controller
        if args.controller == "unet":
            summary["unet_base"] = int(controller_info.get("base", args.unet_base))
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[metrics] saved metrics.json & psnr_curve.npy in {args.save_dir} | controller={args.controller}")
    if log_with_wandb:
    # Finish logging
        wandb.log({
            "final/psnr": summary.get("final_psnr"),
            "final/ssim": summary.get("final_ssim"),
            "final/lpips": summary.get("final_lpips"),
            "final/fid": summary.get("fid"),
            "final/kid": summary.get("kid"),
            "params": param_total,
        })
        wandb.finish()

if __name__ == "__main__":
    main()
