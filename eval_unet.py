"""Evaluate a trained Tiny U-Net on the CIFAR-10 test set."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision.utils import save_image

from image_inpainting import (
    corrupt_images,
    make_transforms,
    random_mask,
    set_seed,
)
from metrics import lpips_dist, param_count, psnr, ssim
from metrics import (fid_init, fid_update, fid_compute,
                     kid_init, kid_update, kid_compute)
from unet_model import TinyUNet


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for Tiny U-Net evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a Tiny U-Net checkpoint on CIFAR-10.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint (.pt) file.")
    parser.add_argument("--batch_size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out_unet_eval",
        help="Directory to store metrics.json and qualitative samples.",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested device string to a :class:`torch.device`."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_checkpoint(ckpt_path: str) -> Dict:
    """Load a checkpoint dictionary from ``ckpt_path``."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint
    # Allow raw state_dict for robustness.
    return {"model": checkpoint}


def extract_base_from_args(args_obj: object, default: int = 10) -> int:
    """Extract the ``base`` attribute/entry from a checkpoint args object."""
    if args_obj is None:
        return default
    if isinstance(args_obj, dict):
        base_val = args_obj.get("base")
        return int(base_val) if base_val is not None else default
    if hasattr(args_obj, "base"):
        base_val = getattr(args_obj, "base")
        if base_val is not None:
            return int(base_val)
    return default


def build_model(checkpoint: Dict, device: torch.device) -> TinyUNet:
    """Instantiate the model and load weights from ``checkpoint``."""
    base = extract_base_from_args(checkpoint.get("args"), default=10)
    model_state = checkpoint.get("model")
    if model_state is None:
        raise KeyError("Checkpoint is missing the 'model' state dictionary.")
    model = TinyUNet(in_ch=4, out_ch=3, base=base)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    return model


def get_dataloader(batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    """Create the CIFAR-10 test dataloader with the standard transforms."""
    transform = make_transforms()
    try:
        dataset = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    except Exception as exc:  # pragma: no cover - offline fallback
        print(f"Dataset download failed ({exc}); retrying without download flag.")
        try:
            dataset = tv.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
        except Exception as inner:  # pragma: no cover - offline fallback
            raise RuntimeError(
                "CIFAR-10 dataset unavailable. Please place the extracted dataset under ./data/cifar-10-batches-py."
            ) from inner
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def masked_psnr(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=a.dtype)
    mask_exp = mask.expand_as(a)
    num = mask_exp.sum().clamp_min(1e-8)
    mse = ((a - b) * mask_exp).pow(2).sum() / num
    if mse <= 0:
        return torch.tensor(99.0, device=a.device, dtype=a.dtype)
    return 10.0 * torch.log10(4.0 / mse)


def masked_metric_mean(
    metric_fn,
    a: Tensor,
    b: Tensor,
    mask: Tensor,
) -> Tensor:
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


def save_samples(
    save_dir: str,
    imgs: Tensor,
    inputs: Tensor,
    masks: Tensor,
    preds: Tensor,
) -> None:
    """Save a qualitative grid of reconstructions for the first batch."""
    os.makedirs(save_dir, exist_ok=True)
    num_show = min(6, imgs.size(0))
    visuals = []
    mask_rgb = masks.repeat(1, 3, 1, 1)
    for idx in range(num_show):
        visuals.extend(
            [
                imgs[idx],
                inputs[idx],
                mask_rgb[idx] * 2.0 - 1.0,
                preds[idx],
            ]
        )
    visual_tensor = torch.stack([(v + 1.0) * 0.5 for v in visuals]).clamp(0.0, 1.0)
    save_path = os.path.join(save_dir, "samples.png")
    save_image(visual_tensor.cpu(), save_path, nrow=4)


def evaluate(model: TinyUNet, loader: DataLoader, device: torch.device, save_dir: str) -> Dict[str, float]:
    """Run evaluation across ``loader`` and return aggregated metrics."""
    metric_names = [
        "psnr_all",
        "psnr_miss",
        "ssim_all",
        "ssim_miss",
        "lpips_all",
        "lpips_miss"
    ]
    totals = {name: 0.0 for name in metric_names}
    batch_count = 0
    fid_metric = fid_init(device)
    kid_metric = kid_init(device)
    sample_saved = False

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            mask_known = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3)
            mask_known = mask_known.to(device)
            inputs = corrupt_images(imgs, mask_known, noise_std=0.3)
            network_input = torch.cat([inputs, mask_known], dim=1)

            preds = model(network_input).clamp(-1.0, 1.0)
            miss_mask = 1.0 - mask_known

            totals["psnr_all"] += psnr(preds, imgs).item()
            totals["psnr_miss"] += masked_psnr(preds, imgs, miss_mask).item()
            totals["ssim_all"] += ssim(preds, imgs).item()
            totals["ssim_miss"] += masked_metric_mean(ssim, preds, imgs, miss_mask).item()
            totals["lpips_all"] += lpips_dist(preds, imgs).item()
            totals["lpips_miss"] += masked_metric_mean(lpips_dist, preds, imgs, miss_mask).item()
            # Accumulate FID/KID features
            fid_update(fid_metric, imgs, preds)
            kid_update(kid_metric, imgs, preds)
            batch_count += 1

            if not sample_saved:
                save_samples(save_dir, imgs, inputs, mask_known, preds)
                sample_saved = True

    if batch_count == 0:
        raise RuntimeError("Evaluation loader produced no batches.")

    results = {name: totals[name] / batch_count for name in metric_names}
    # Compute final FID/KID scores over full test set
    results["fid"] = fid_compute(fid_metric)
    results["kid"] = kid_compute(kid_metric)
    return results


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    set_seed(args.seed)

    checkpoint = load_checkpoint(args.ckpt)
    model = build_model(checkpoint, device)
    params = param_count(model)

    loader = get_dataloader(args.batch_size, args.num_workers, device)
    os.makedirs(args.save_dir, exist_ok=True)

    metrics = evaluate(model, loader, device, args.save_dir)
    metrics["params"] = params
    metrics["seed"] = args.seed

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    print(f"Parameters: {params:,}")
    print(f"Seed: {args.seed}")
    print("Metrics:")
    for key in [
        "psnr_all",
        "psnr_miss",
        "ssim_all",
        "ssim_miss",
        "lpips_all",
        "lpips_miss",
        "fid",
        "kid"
    ]:
        print(f"  {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
