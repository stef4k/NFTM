"""Utility metrics for image quality and model inspection.

This module provides PSNR, SSIM, and LPIPS helpers tailored for tensors
in the range ``[-1, 1]`` with shape ``(B, 3, H, W)``. It also offers a
simple trainable parameter counting utility.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:  # Prefer pytorch-msssim when available.
    from pytorch_msssim import ssim as _msssim

    _SSIM_BACKEND = "pytorch-msssim"
except Exception:  # pragma: no cover - optional dependency
    _msssim = None
    try:
        from torchmetrics.functional.image.ssim import (
            structural_similarity_index_measure as _tm_ssim,
        )

        _SSIM_BACKEND = "torchmetrics"
    except Exception:  # pragma: no cover - optional dependency
        _tm_ssim = None
        _SSIM_BACKEND = "fallback"


__all__ = ["psnr", "ssim", "lpips_dist", "param_count"]


_LPIPS_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
_SSIM_KERNEL_CACHE: Dict[Tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}


def _validate_inputs(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate input tensors and cast them to ``float32`` for computation."""
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor instances.")
    if x.shape != y.shape:
        raise ValueError(f"Input shapes must match, got {x.shape} and {y.shape}.")
    if x.ndim != 4 or x.size(1) != 3:
        raise ValueError("Inputs must have shape (B, 3, H, W).")
    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise TypeError("Inputs must be floating point tensors.")

    x32 = x.detach().to(dtype=torch.float32)
    y32 = y.detach().to(dtype=torch.float32)

    if torch.isnan(x32).any() or torch.isnan(y32).any():
        raise ValueError("Inputs contain NaNs, which is not supported.")

    if (x32 < -1.0).any() or (x32 > 1.0).any():
        raise ValueError("Input x must be within [-1, 1].")
    if (y32 < -1.0).any() or (y32 > 1.0).any():
        raise ValueError("Input y must be within [-1, 1].")

    return x32, y32


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the batch-mean PSNR (dB) for tensors in ``[-1, 1]``."""

    x32, y32 = _validate_inputs(x, y)
    device = x.device

    diff = (x32 - y32).to(dtype=torch.float64)
    mse = torch.mean(diff * diff)
    if mse <= 0:
        return torch.tensor(99.0, device=device, dtype=torch.float32)

    max_val = torch.tensor(2.0, dtype=torch.float64, device=diff.device)
    psnr_value = 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse)
    return psnr_value.to(device=device, dtype=torch.float32)


def _get_gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a separable 2D Gaussian kernel expanded for ``channels`` groups."""

    key = (window_size, sigma, device, dtype)
    kernel = _SSIM_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d = gauss_2d / gauss_2d.sum()
    kernel = gauss_2d.expand(channels, 1, window_size, window_size).contiguous()
    _SSIM_KERNEL_CACHE[key] = kernel
    return kernel


def _ssim_fallback(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute SSIM using a pure PyTorch implementation."""

    channels = x.size(1)
    height, width = x.size(2), x.size(3)
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        window_size = 3
    sigma = 1.5 * window_size / 11.0

    kernel = _get_gaussian_kernel(
        window_size, sigma, channels, device=x.device, dtype=x.dtype
    )
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    data_range = torch.tensor(2.0, dtype=x.dtype, device=x.device)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.mean()


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the batch-mean SSIM for tensors in ``[-1, 1]``."""

    x32, y32 = _validate_inputs(x, y)
    device = x.device

    if _SSIM_BACKEND == "pytorch-msssim":
        ssim_val = _msssim(x32, y32, data_range=2.0, size_average=True, win_size=11)
    elif _SSIM_BACKEND == "torchmetrics":
        ssim_val = _tm_ssim(x32, y32, data_range=2.0, kernel_size=11)
    else:
        ssim_val = _ssim_fallback(x32, y32)

    if not isinstance(ssim_val, torch.Tensor):  # pragma: no cover - defensive
        ssim_val = torch.tensor(ssim_val, device=device, dtype=torch.float32)

    return ssim_val.to(device=device, dtype=torch.float32)


def lpips_dist(x: torch.Tensor, y: torch.Tensor, net: str = "alex") -> torch.Tensor:
    """Return the batch-mean LPIPS distance between ``x`` and ``y``."""

    if net not in {"alex", "vgg", "squeeze"}:
        raise ValueError("'net' must be one of {'alex', 'vgg', 'squeeze'}.")

    x32, y32 = _validate_inputs(x, y)
    device = x.device

    key = (str(device), net)
    model = _LPIPS_CACHE.get(key)

    if model is None:
        try:
            import lpips
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("The 'lpips' package is required for lpips_dist().") from exc

        model = lpips.LPIPS(net=net)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    else:
        if next(model.parameters()).device != device:  # pragma: no cover - defensive
            model = model.to(device)
            _LPIPS_CACHE[key] = model

    with torch.inference_mode():
        dist = model(x32.to(device), y32.to(device))

    dist = dist.to(device=device, dtype=torch.float32)
    return dist.mean()


def param_count(model) -> int:
    """Return the number of trainable parameters in ``model``."""

    if model is None:
        raise ValueError("model must not be None")
    if not hasattr(model, "parameters"):
        raise TypeError("model must expose a parameters() iterator")

    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    return int(total)


if __name__ == "__main__":  # pragma: no cover - minimal runtime checks
    torch.manual_seed(0)
    cpu_device = torch.device("cpu")
    x_cpu = torch.rand(8, 3, 32, 32, device=cpu_device) * 2 - 1
    y_cpu = torch.rand(8, 3, 32, 32, device=cpu_device) * 2 - 1

    print("CPU metrics:")
    print("  PSNR:", psnr(x_cpu, y_cpu).item())
    try:
        print("  SSIM:", ssim(x_cpu, y_cpu).item())
    except RuntimeError as exc:
        print("  SSIM unavailable:", exc)
    try:
        print("  LPIPS:", lpips_dist(x_cpu, y_cpu).item())
    except RuntimeError as exc:
        print("  LPIPS unavailable:", exc)

    print("Identity checks on CPU:")
    print("  PSNR(x,x):", psnr(x_cpu, x_cpu).item())
    try:
        ssim_xx = ssim(x_cpu, x_cpu).item()
        print("  SSIM(x,x):", ssim_xx)
        assert 0.0 <= ssim_xx <= 1.0
    except RuntimeError as exc:
        print("  SSIM unavailable:", exc)
    try:
        lpips_xx = lpips_dist(x_cpu, x_cpu).item()
        print("  LPIPS(x,x):", lpips_xx)
        assert lpips_xx <= 1e-4
    except RuntimeError as exc:
        print("  LPIPS unavailable:", exc)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        x_cuda = x_cpu.to(device)
        y_cuda = y_cpu.to(device)
        print("CUDA metrics:")
        print("  PSNR:", psnr(x_cuda, y_cuda).item())
        try:
            print("  SSIM:", ssim(x_cuda, y_cuda).item())
        except RuntimeError as exc:
            print("  SSIM unavailable:", exc)
        try:
            print("  LPIPS:", lpips_dist(x_cuda, y_cuda).item())
        except RuntimeError as exc:
            print("  LPIPS unavailable:", exc)

        print("Identity checks on CUDA:")
        print("  PSNR(x,x):", psnr(x_cuda, x_cuda).item())
        try:
            ssim_xx = ssim(x_cuda, x_cuda).item()
            print("  SSIM(x,x):", ssim_xx)
            assert 0.0 <= ssim_xx <= 1.0
        except RuntimeError as exc:
            print("  SSIM unavailable:", exc)
        try:
            lpips_xx = lpips_dist(x_cuda, x_cuda).item()
            print("  LPIPS(x,x):", lpips_xx)
            assert lpips_xx <= 1e-4
        except RuntimeError as exc:
            print("  LPIPS unavailable:", exc)
