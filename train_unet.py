#!/usr/bin/env python3
"""Training script for the Tiny U-Net inpainting baseline on CIFAR-10."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision as tv
from torchvision.utils import make_grid, save_image

from image_inpainting import (
    corrupt_images,
    get_transform,
    random_mask,
    set_seed,
    tv_l1,
)
from metrics import lpips_dist, param_count, ssim
from unet_model import TinyUNet


@dataclass
class TrainRecord:
    epoch: int
    loss: float
    psnr_val: Optional[float]
    time_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny U-Net on CIFAR-10 inpainting")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="out_unet")
    parser.add_argument("--target_params", type=int, default=46375)
    parser.add_argument("--base", type=int, default=10, help="Base channel width for TinyUNet")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--val_batches", type=int, default=8, help="Number of validation batches per epoch")
    parser.add_argument("--val_size", type=int, default=2000, help="Number of validation images")
    parser.add_argument("--data_root", type=str, default="data", help="Path to the CIFAR-10 dataset root")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def create_dataloaders(
    batch_size: int,
    num_workers: int,
    val_size: int,
    data_root: str,
) -> Tuple[DataLoader, DataLoader, torch.utils.data.Dataset]:
    transform = get_transform()
    train_ds = load_cifar_split(data_root, True, transform)
    test_ds = load_cifar_split(data_root, False, transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if val_size > 0:
        val_indices = list(range(min(val_size, len(test_ds))))
        val_subset = Subset(test_ds, val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_ds


def load_cifar_split(root: str, train: bool, transform) -> torch.utils.data.Dataset:
    try:
        return tv.datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    except RuntimeError as exc:
        if "download=True" not in str(exc):
            raise
        try:
            return tv.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        except Exception as download_exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "Failed to download CIFAR-10 dataset. Please ensure it is available at"
                f" '{root}' or provide network access."
            ) from download_exc


def masked_psnr_components(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    mask_b = mask.expand_as(target)
    diff = (pred - target) ** 2 * mask_b
    mse_sum = diff.sum().item()
    denom = mask_b.sum().item()
    return mse_sum, max(denom, 1e-8)


def evaluate_psnr(
    model: TinyUNet,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_se = 0.0
    total_pixels = 0.0
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            mask = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3)
            corrupted = corrupt_images(imgs, mask, noise_std=0.3)
            inp = torch.cat([corrupted, mask], dim=1)
            pred = model(inp)
            diff = pred - imgs
            total_se += diff.pow(2).sum().item()
            total_pixels += diff.numel()
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    mse = total_se / max(total_pixels, 1e-8)
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(4.0 / mse)


def train_one_epoch(
    model: TinyUNet,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    tv_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        mask = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3)
        corrupted = corrupt_images(imgs, mask, noise_std=0.3)
        inp = torch.cat([corrupted, mask], dim=1)

        pred = model(inp)
        data_loss = F.l1_loss(pred, imgs)
        tv_loss = tv_l1(pred)
        loss = data_loss + tv_weight * tv_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate_full(
    model: TinyUNet,
    loader: DataLoader,
    device: torch.device,
    sample_path: Optional[str] = None,
    sample_rows: int = 6,
) -> Dict[str, float]:
    model.eval()
    total_mse_all = 0.0
    total_pixels_all = 0.0
    total_mse_miss = 0.0
    total_pixels_miss = 0.0
    total_ssim_all = 0.0
    total_ssim_miss = 0.0
    total_lpips_all = 0.0
    total_lpips_miss = 0.0
    total_images = 0
    sample_captured = False

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            mask = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3)
            corrupted = corrupt_images(imgs, mask, noise_std=0.3)
            inp = torch.cat([corrupted, mask], dim=1)
            pred = model(inp)

            diff = pred - imgs
            total_mse_all += diff.pow(2).sum().item()
            total_pixels_all += diff.numel()

            miss_mask = 1.0 - mask
            se_miss, denom_miss = masked_psnr_components(pred, imgs, miss_mask)
            total_mse_miss += se_miss
            total_pixels_miss += denom_miss

            mask_b = miss_mask.expand_as(imgs)
            ssim_all = ssim(pred, imgs).item()
            ssim_miss = ssim(pred * mask_b, imgs * mask_b).item()
            lpips_all = lpips_dist(pred, imgs).item()
            lpips_miss = lpips_dist(pred * mask_b, imgs * mask_b).item()

            bsz = imgs.size(0)
            total_ssim_all += ssim_all * bsz
            total_ssim_miss += ssim_miss * bsz
            total_lpips_all += lpips_all * bsz
            total_lpips_miss += lpips_miss * bsz
            total_images += bsz

            if sample_path and not sample_captured:
                sample_captured = True
                save_prediction_grid(imgs, corrupted, mask, pred, sample_path, sample_rows)

    mse_all = total_mse_all / max(total_pixels_all, 1e-8)
    mse_miss = total_mse_miss / max(total_pixels_miss, 1e-8)
    psnr_all = 10.0 * math.log10(4.0 / max(mse_all, 1e-12)) if mse_all > 0 else 99.0
    psnr_miss = 10.0 * math.log10(4.0 / max(mse_miss, 1e-12)) if mse_miss > 0 else 99.0

    metrics = {
        "psnr_all": psnr_all,
        "psnr_miss": psnr_miss,
        "ssim_all": total_ssim_all / max(total_images, 1),
        "ssim_miss": total_ssim_miss / max(total_images, 1),
        "lpips_all": total_lpips_all / max(total_images, 1),
        "lpips_miss": total_lpips_miss / max(total_images, 1),
    }
    return metrics


def save_prediction_grid(
    gt: torch.Tensor,
    corrupted: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    path: str,
    rows: int,
) -> None:
    rows = min(rows, gt.size(0))
    images: List[torch.Tensor] = []
    mask_vis = mask.repeat(1, 3, 1, 1) * 2.0 - 1.0
    for i in range(rows):
        images.append(gt[i].detach().cpu())
        images.append(corrupted[i].detach().cpu())
        images.append(mask_vis[i].detach().cpu())
        images.append(pred[i].detach().cpu())
    grid = make_grid(images, nrow=4, padding=2, normalize=True, value_range=(-1, 1))
    save_image(grid, path)


def plot_psnr_curve(psnr_values: List[Optional[float]], save_path: str) -> None:
    epochs = [i + 1 for i, v in enumerate(psnr_values) if v is not None]
    values = [v for v in psnr_values if v is not None]
    if not values:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_ds = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        data_root=args.data_root,
    )

    model = TinyUNet(in_ch=4, out_ch=3, base=args.base).to(device)
    params = param_count(model)
    print(f"Model parameters: {params}")
    if args.target_params > 0:
        delta = abs(params - args.target_params) / args.target_params
        if delta > 0.05:
            print(
                f"[Warning] Parameter count {params} deviates more than 5% from target {args.target_params}."
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    val_psnr_history: List[Optional[float]] = []
    log_records: List[Dict[str, object]] = []
    best_val_psnr = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, args.tv_weight)

        val_psnr = None
        if len(val_loader) > 0:
            val_psnr = evaluate_psnr(model, val_loader, device, max_batches=args.val_batches)
            val_psnr_history.append(val_psnr)
        else:
            val_psnr_history.append(None)

        elapsed = time.time() - start_time
        record = TrainRecord(epoch=epoch, loss=train_loss, psnr_val=val_psnr, time_sec=elapsed)
        log_records.append(asdict(record))

        if val_psnr is not None and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d}: loss={train_loss:.4f}"
            + (f", val_psnr={val_psnr:.2f} dB" if val_psnr is not None else "")
            + f", time={elapsed:.1f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    full_test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample_path = os.path.join(args.save_dir, "pred_samples.png")
    metrics = evaluate_full(model, full_test_loader, device, sample_path=sample_path)
    metrics.update({"params": params, "seed": args.seed})

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "params": params,
        "best_metric": metrics.get("psnr_all", float("nan")),
    }
    torch.save(ckpt, os.path.join(args.save_dir, "ckpt.pt"))

    with open(os.path.join(args.save_dir, "train_log.json"), "w") as f:
        json.dump(log_records, f, indent=2)

    plot_psnr_curve(val_psnr_history, os.path.join(args.save_dir, "psnr_curve.png"))

    print("Training and evaluation complete.")


if __name__ == "__main__":
    main()
