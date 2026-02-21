#!/usr/bin/env python3
# data_and_viz.py

import os, random, argparse, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

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

def plot_metric_curve(curve, save_path, ylabel, title):
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(6,4))
    x = np.arange(len(curve))
    plt.plot(x, curve, marker='o')
    plt.xlim(left=0)
    plt.xlabel("NFTM step"); plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=160)
    print(f"[plot] saved {title.lower()} → {save_path}")
    plt.close()
