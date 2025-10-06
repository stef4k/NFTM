"""Shared utilities for heat-equation style scalar-field metrics and timing.

This module consolidates helpers that were previously duplicated across the
baselines.  All utilities operate on scalar fields shaped ``(B, T, 1, H, W)``
or ``(B, 1, H, W)`` and return python ``float`` values for easy logging.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import torch

def _ensure_rollout_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the tensor has ``(B, T, 1, H, W)`` dimensions."""
    if tensor.ndim == 4:  # (B, 1, H, W)
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5 or tensor.size(2) != 1:
        raise ValueError(
            "Expected tensor shaped (B, T, 1, H, W) or (B, 1, H, W); "
            f"got {tuple(tensor.shape)}."
        )
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor


def _broadcast_mask(mask: Optional[torch.Tensor], target_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
    """Validate and broadcast a mask to ``target_shape`` if provided."""
    if mask is None:
        return None
    mask = _ensure_rollout_dims(mask)
    if mask.shape != target_shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match tensor shape {target_shape}.")
    return mask
def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    diff = pred - target
    if mask is None:
        return torch.mean(diff * diff)
    weight = mask.to(dtype=pred.dtype)
    total_weight = torch.sum(weight)
    if total_weight <= 0:
        raise ValueError("Mask must include at least one valid element for PSNR computation.")
    return torch.sum(weight * diff * diff) / total_weight


def _masked_min_max(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return tensor.amin(), tensor.amax()
    weight = mask.to(dtype=tensor.dtype)
    if torch.sum(weight) <= 0:
        raise ValueError("Mask must include at least one valid element for range computation.")
    masked_vals = tensor[weight > 0]
    return masked_vals.amin(), masked_vals.amax()


def scalar_rollout_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
) -> float:
    """Compute PSNR aggregated over an entire rollout."""
    pred = _ensure_rollout_dims(pred)
    target = _ensure_rollout_dims(target)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes must match, got {tuple(pred.shape)} and {tuple(target.shape)}.")
    mask = _broadcast_mask(mask, target.shape)

    with torch.no_grad():
        mse = _masked_mse(pred, target, mask)
        if mse <= 0:
            return float("inf")
        if data_range is None:
            min_val, max_val = _masked_min_max(target, mask)
            data_range_value = float((max_val - min_val).item())
        else:
            data_range_value = float(data_range)
        if data_range_value <= 0:
            return float("inf")
        data_range_tensor = torch.tensor(data_range_value, device=pred.device, dtype=pred.dtype)
        psnr_val = 20.0 * torch.log10(data_range_tensor) - 10.0 * torch.log10(mse)
        return float(psnr_val.item())


def _weighted_mean(tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    total_weight = torch.sum(weight)
    if total_weight <= 0:
        raise ValueError("Weight tensor must include positive mass.")
    return torch.sum(tensor * weight) / total_weight


def _weighted_ssim_single(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    data_range: float,
) -> torch.Tensor:
    if mask is None:
        weight = torch.ones_like(pred)
    else:
        weight = mask.to(dtype=pred.dtype)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = _weighted_mean(pred, weight)
    mu_y = _weighted_mean(target, weight)
    var_x = _weighted_mean((pred - mu_x) ** 2, weight)
    var_y = _weighted_mean((target - mu_y) ** 2, weight)
    cov_xy = _weighted_mean((pred - mu_x) * (target - mu_y), weight)
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    if denominator <= 0:
        return torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
    return numerator / denominator


def scalar_rollout_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
) -> float:
    """Compute SSIM aggregated over an entire rollout using a global formulation."""
    pred = _ensure_rollout_dims(pred)
    target = _ensure_rollout_dims(target)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes must match, got {tuple(pred.shape)} and {tuple(target.shape)}.")
    mask = _broadcast_mask(mask, target.shape)

    with torch.no_grad():
        if data_range is None:
            min_val, max_val = _masked_min_max(target, mask)
            data_range_value = float((max_val - min_val).item())
        else:
            data_range_value = float(data_range)
        if data_range_value <= 0:
            return 1.0
        pred_flat = pred.reshape(-1, pred.size(-2) * pred.size(-1))
        target_flat = target.reshape(-1, target.size(-2) * target.size(-1))
        if mask is not None:
            mask_flat = mask.reshape(-1, mask.size(-2) * mask.size(-1))
        else:
            mask_flat = None
        scores = []
        for idx in range(pred_flat.size(0)):
            current_mask = None if mask_flat is None else mask_flat[idx]
            scores.append(
                _weighted_ssim_single(
                    pred_flat[idx],
                    target_flat[idx],
                    current_mask,
                    data_range=data_range_value,
                )
            )
        mean_score = torch.stack(scores).mean()
        return float(mean_score.item())


@dataclass
class _TimerStats:
    total_time: float = 0.0
    count: int = 0

    def update(self, duration: float) -> None:
        self.total_time += duration
        self.count += 1

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_time / self.count


class _TimerContext:
    def __init__(self, stats: _TimerStats):
        self._stats = stats
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is not None and exc_type is None:
            duration = time.perf_counter() - self._start
            self._stats.update(duration)
        self._start = None
        return False


class TrainTimer:
    """Measure average epoch times via a context manager."""

    def __init__(self) -> None:
        self._stats = _TimerStats()

    def record_epoch(self) -> _TimerContext:
        return _TimerContext(self._stats)

    @property
    def average_epoch_time(self) -> float:
        return self._stats.average


class InferenceTimer:
    """Measure average inference step time via a context manager."""

    def __init__(self) -> None:
        self._stats = _TimerStats()

    def record_step(self) -> _TimerContext:
        return _TimerContext(self._stats)

    @property
    def average_step_time(self) -> float:
        return self._stats.average


def compute_rollout_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute a dictionary of scalar-field metrics with consistent keys."""
    metrics: Dict[str, float] = {}
    key_prefix = f"{prefix}/" if prefix else ""
    metrics[f"{key_prefix}psnr"] = scalar_rollout_psnr(pred, target, None)
    metrics[f"{key_prefix}ssim"] = scalar_rollout_ssim(pred, target, None)
    if mask is not None:
        metrics[f"{key_prefix}psnr_masked"] = scalar_rollout_psnr(pred, target, mask)
        metrics[f"{key_prefix}ssim_masked"] = scalar_rollout_ssim(pred, target, mask)
    return metrics


def dump_metrics(path: Path, metrics: Mapping[str, float]) -> None:
    """Write ``metrics`` to ``path`` as ``metrics.json`` compatible output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable: MutableMapping[str, float] = {key: float(value) for key, value in metrics.items()}
    path.write_text(json.dumps(serialisable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "TrainTimer",
    "InferenceTimer",
    "compute_rollout_metrics",
    "dump_metrics",
    "scalar_rollout_psnr",
    "scalar_rollout_ssim",
]
