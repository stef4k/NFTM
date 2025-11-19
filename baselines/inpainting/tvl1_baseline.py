"""TV-L1 inpainting baseline using a primal-dual solver."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision as tv
from torchvision.utils import save_image
from nftm_inpaint.metrics import (fid_init, fid_update, fid_compute,
                     kid_init, kid_update, kid_compute)

root_dir = os.path.dirname(os.path.abspath(__file__))  # directory containing this file
grandparent_dir = os.path.abspath(os.path.join(root_dir, "..", ".."))  # parent of parent
benchmarks_dir = os.path.join(grandparent_dir, "benchmarks")

try:  # Prefer shared helpers if available
    from nftm_inpaint.rollout import (
        clamp_known,
        corrupt_images 
    )
    from nftm_inpaint.data_and_viz import set_seed, get_transform, random_mask
except ImportError:  # pragma: no cover - fallback copies (should not trigger in repo)
    import random

    import numpy as np
    import torchvision.transforms as T

    def set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def random_mask(batch, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3):
        B, C, H, W = batch.shape
        device = batch.device
        M = torch.ones((B, 1, H, W), device=device)
        frac = torch.empty(B, 1, 1, 1, device=device).uniform_(*p_missing)
        pix_mask = (torch.rand(B, 1, H, W, device=device) > frac).float()
        M = M * pix_mask
        for b in range(B):
            if random.random() < block_prob:
                for _ in range(random.randint(min_blocks, max_blocks)):
                    sz = random.randint(H // 8, H // 3)
                    y = random.randint(0, H - sz)
                    x = random.randint(0, W - sz)
                    M[b, :, y : y + sz, x : x + sz] = 0.0
        return M

    def corrupt_images(img, M, noise_std=0.3):
        noise = torch.randn_like(img) * noise_std
        return M * img + (1 - M) * noise

    def clamp_known(I, I_gt, M):
        return I * (1 - M) + I_gt * M

try:
    from metrics import lpips_dist as _metrics_lpips
    from metrics import psnr as _metrics_psnr
    from metrics import ssim as _metrics_ssim
except ImportError:  # Provide lightweight fallbacks if metrics module missing

    def _psnr(a: torch.Tensor, b: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        eps = 1e-8
        diff = (a - b).flatten(start_dim=1)
        mse = diff.pow(2).mean(dim=1).clamp_min(eps)
        val = 10.0 * torch.log10(4.0 / mse)
        if reduction == "none":
            return val
        return val.mean()

    def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = (g / g.sum()).unsqueeze(0)
        kernel = (g.t() @ g).unsqueeze(0).unsqueeze(0)
        return kernel

    _ssim_cache: Dict[tuple, torch.Tensor] = {}

    def _ssim(a: torch.Tensor, b: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        B, C, H, W = a.shape
        device, dtype = a.device, a.dtype
        window_size = 11
        sigma = 1.5
        key = (window_size, sigma, device, dtype)
        if key not in _ssim_cache:
            kernel = _gaussian_kernel(window_size, sigma, device, dtype)
            kernel = kernel.expand(C, 1, window_size, window_size)
            _ssim_cache[key] = kernel
        kernel = _ssim_cache[key]
        pad = window_size // 2
        a01 = (a + 1.0) * 0.5
        b01 = (b + 1.0) * 0.5
        mu_a = F.conv2d(a01, kernel, padding=pad, groups=C)
        mu_b = F.conv2d(b01, kernel, padding=pad, groups=C)
        mu_a2 = mu_a.pow(2)
        mu_b2 = mu_b.pow(2)
        mu_ab = mu_a * mu_b
        sigma_a2 = F.conv2d(a01 * a01, kernel, padding=pad, groups=C) - mu_a2
        sigma_b2 = F.conv2d(b01 * b01, kernel, padding=pad, groups=C) - mu_b2
        sigma_ab = F.conv2d(a01 * b01, kernel, padding=pad, groups=C) - mu_ab
        c1 = (0.01 * 2.0) ** 2
        c2 = (0.03 * 2.0) ** 2
        ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
        val = ssim_map.flatten(start_dim=1).mean(dim=1)
        if reduction == "none":
            return val
        return val.mean()

    def _lpips(a: torch.Tensor, b: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        def _normalize(x: torch.Tensor) -> torch.Tensor:
            return x / torch.sqrt(x.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-6)

        feats_a, feats_b = a, b
        dists = []
        for _ in range(3):
            na, nb = _normalize(feats_a), _normalize(feats_b)
            dists.append((na - nb).pow(2).flatten(start_dim=1).mean(dim=1))
            if feats_a.shape[-1] <= 4 or feats_a.shape[-2] <= 4:
                break
            feats_a = F.avg_pool2d(feats_a, kernel_size=2, stride=2, padding=0)
            feats_b = F.avg_pool2d(feats_b, kernel_size=2, stride=2, padding=0)
        dist = torch.stack(dists).mean(dim=0)
        if reduction == "none":
            return dist
        return dist.mean()

    _metrics_psnr = _psnr
    _metrics_ssim = _ssim
    _metrics_lpips = _lpips


@dataclass
class TVL1Params:
    iters: int = 400
    lam: float = 40.0
    tvw: float = 0.05
    tau_p: float = 0.25
    tau_d: float = 0.25


@torch.no_grad()
def tvl1_inpaint(
    I0: torch.Tensor,
    M: torch.Tensor,
    iters: int = 400,
    lam: float = 40.0,
    tvw: float = 0.05,
    tau_p: float = 0.25,
    tau_d: float = 0.25,
) -> torch.Tensor:
    """Batched anisotropic TV-L1 inpainting via a primal-dual solver."""

    if I0.ndim != 4 or I0.size(1) != 3:
        raise ValueError("I0 must have shape (B,3,H,W)")
    if M.ndim != 4 or M.size(1) != 1:
        raise ValueError("M must have shape (B,1,H,W)")
    if I0.shape[0] != M.shape[0] or I0.shape[2:] != M.shape[2:]:
        raise ValueError("I0 and M must share batch and spatial dims")

    device = I0.device
    dtype = I0.dtype
    B, C, H, W = I0.shape
    mask = M.to(dtype=dtype, device=device)
    mask_exp = mask.expand(-1, C, -1, -1)

    I = I0.clone()
    px = torch.zeros_like(I)
    py = torch.zeros_like(I)
    grad_x = torch.zeros_like(I)
    grad_y = torch.zeros_like(I)
    div = torch.zeros_like(I)

    tau_p = float(tau_p)
    tau_d = float(tau_d)
    lam = float(lam)
    tvw = float(tvw)
    for _ in range(int(iters)):
        grad_x.zero_()
        grad_y.zero_()
        grad_x[:, :, :, :-1].copy_(I[:, :, :, 1:] - I[:, :, :, :-1])
        grad_y[:, :, :-1, :].copy_(I[:, :, 1:, :] - I[:, :, :-1, :])

        px.add_(grad_x, alpha=tau_d)
        py.add_(grad_y, alpha=tau_d)
        px[:, :, :, -1].zero_()
        py[:, :, -1, :].zero_()

        px = px / torch.clamp(px.abs() / tvw, min=1.0)
        py = py / torch.clamp(py.abs() / tvw, min=1.0)
        px[:, :, :, -1].zero_()
        py[:, :, -1, :].zero_()

        div.zero_()
        div.add_(px)
        div[:, :, :, 1:].sub_(px[:, :, :, :-1])
        div.add_(py)
        div[:, :, 1:, :].sub_(py[:, :, :-1, :])

        data_term = lam * mask_exp * (I - I0)
        I.add_(div, alpha=tau_p)  # + τ div p
        I.add_(data_term, alpha=-tau_p)  # − τ λ M (u − f)
        I = clamp_known(I, I0, mask)
        I.clamp_(-1.0, 1.0)

    I = clamp_known(I, I0, mask)
    return I


def masked_psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    num = mask_exp.sum().clamp_min(1e-8)
    mse = ((a - b) * mask_exp).pow(2).sum() / num
    if mse <= 0:
        return torch.tensor(99.0, device=a.device, dtype=a.dtype)
    return 10.0 * torch.log10(4.0 / mse)


def masked_metric_mean(
    metric_fn: Callable[..., torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    try:
        vals = metric_fn(a * mask_exp, b * mask_exp, reduction="none")
        return vals.mean()
    except TypeError:
        val = metric_fn(a * mask_exp, b * mask_exp)
        return torch.as_tensor(val, device=a.device, dtype=a.dtype)


def run_eval(params: TVL1Params, args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    transform_test = get_transform(args.benchmark, img_size=args.img_size)
    # Set test dataset
    if args.benchmark == "cifar":
        dataset = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif args.benchmark == "set12":
        dataset = ImageFolder(root=os.path.join(benchmarks_dir, "Set12"), transform=transform_test)
    elif args.benchmark == "cbsd68":
        dataset = ImageFolder(root=os.path.join(benchmarks_dir,"CBSD68"), transform=transform_test)
    elif args.benchmark == "celebahq":
        train_set = ImageFolder(root=os.path.join(benchmarks_dir, "CelebAHQ"), transform=transform_test)
        train_size = int(0.8 * len(train_set))
        test_size = len(train_set) - train_size
        _, dataset = random_split(train_set, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError(f"Unknown benchmark dataset: {args.benchmark}")
    
    print(f"[TVL1 Eval Data] benchmark={args.benchmark}, img_size={args.img_size}")
    print(f"[TVL1 Eval Data] dataset size={len(dataset)}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(args.save_dir, exist_ok=True)

    metrics_sum = {
        "psnr_all": 0.0,
        "psnr_miss": 0.0,
        "ssim_all": 0.0,
        "ssim_miss": 0.0,
        "lpips_all": 0.0,
        "lpips_miss": 0.0,
    }
    batch_count = 0
    fid_metric = fid_init(device)
    kid_metric = kid_init(device)
    sample_saved = False

    for imgs, _ in loader:
        imgs = imgs.to(device)
        M = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3).to(device)
        I0 = corrupt_images(imgs, M, noise_std=0.3)
        I_hat = tvl1_inpaint(
            I0,
            M,
            iters=params.iters,
            lam=params.lam,
            tvw=params.tvw,
            tau_p=params.tau_p,
            tau_d=params.tau_d,
        )

        miss = 1.0 - M
        psnr_all = _metrics_psnr(I_hat, imgs).item()
        psnr_miss = masked_psnr(I_hat, imgs, miss).item()
        ssim_all = _metrics_ssim(I_hat, imgs).item()
        ssim_miss = masked_metric_mean(_metrics_ssim, I_hat, imgs, miss).item()
        lpips_all = _metrics_lpips(I_hat, imgs).item()
        lpips_miss = masked_metric_mean(_metrics_lpips, I_hat, imgs, miss).item()

        metrics_sum["psnr_all"] += psnr_all
        metrics_sum["psnr_miss"] += psnr_miss
        metrics_sum["ssim_all"] += ssim_all
        metrics_sum["ssim_miss"] += ssim_miss
        metrics_sum["lpips_all"] += lpips_all
        metrics_sum["lpips_miss"] += lpips_miss
        # Accumulate FID/KID features
        fid_update(fid_metric, imgs, I_hat)
        kid_update(kid_metric, imgs, I_hat)
        batch_count += 1

        if not sample_saved:
            num_show = min(6, imgs.size(0))
            visuals = []
            miss_rgb = miss.repeat(1, 3, 1, 1)
            for idx in range(num_show):
                visuals.extend(
                    [
                        imgs[idx],
                        I0[idx],
                        (miss_rgb[idx] * 2.0) - 1.0,
                        I_hat[idx],
                    ]
                )
            visual_tensor = torch.stack([(v + 1.0) * 0.5 for v in visuals]).clamp(0.0, 1.0)
            save_image(visual_tensor.cpu(), os.path.join(args.save_dir, "samples.png"), nrow=4)
            sample_saved = True

    for key in metrics_sum:
        metrics_sum[key] = metrics_sum[key] / max(batch_count, 1)
    # Compute final FID/KID scores over full test set
    metrics_sum["fid"] = fid_compute(fid_metric)
    metrics_sum["kid"] = kid_compute(kid_metric)

    metrics_sum["seed"] = args.seed
    print(
        f"[TVL1] mean PSNR_all={metrics_sum['psnr_all']:.2f} dB, "
        f"PSNR_miss={metrics_sum['psnr_miss']:.2f} dB"
    )
    return metrics_sum


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TV-L1 inpainting baseline")
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--lam", type=float, default=40.0)
    parser.add_argument("--tvw", type=float, default=0.05)
    parser.add_argument("--tau_p", type=float, default=0.25)
    parser.add_argument("--tau_d", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_dir", type=str, default="out_tvl1")
    parser.add_argument("--no_cuda", action="store_true", help="Deprecated flag (ignored)")
    parser.add_argument("--benchmark", type=str, default="cifar", help="Dataset key for choosing transforms")
    parser.add_argument("--img_size", type=int, default=32, choices=[32, 64], help="Input image size (resize if necessary)")

    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    args.device = resolve_device(args.device)
    set_seed(args.seed)

    params = TVL1Params(
        iters=args.iters,
        lam=args.lam,
        tvw=args.tvw,
        tau_p=args.tau_p,
        tau_d=args.tau_d,
    )

    print(
        f"Running TV-L1 baseline on {args.benchmark} test set with {params.iters} iterations "
        f"(device={args.device})"
    )
    metrics = run_eval(params, args)

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
