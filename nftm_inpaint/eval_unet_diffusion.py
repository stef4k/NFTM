"""Evaluate a trained Tiny U-Net Diffusion checkpoint on the test set."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from data_and_viz import (random_mask, set_seed, get_transform)
from nftm_inpaint.metrics import lpips_dist, param_count, psnr, ssim
from nftm_inpaint.metrics import (fid_init, fid_update, fid_compute,
                     kid_init, kid_update, kid_compute)
from nftm_inpaint.unet_model import BigUNetDiffusion, TinyUNetDiffusion

root_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.abspath(os.path.join(root_dir, ".."))  
benchmarks_dir = os.path.join(parent_dir, "benchmarks")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for Tiny U-Net diffusion evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a Tiny U-Net Diffusion checkpoint.")
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
    parser.add_argument("--benchmark", type=str, default="cifar", help="Dataset key for choosing transforms")
    parser.add_argument("--img_size", type=int, default=32, choices=[32, 64, 128], help="Input image size (resize if necessary)")
    parser.add_argument("--beta_start", type=float, default=None, help="Override diffusion beta_start")
    parser.add_argument("--beta_end", type=float, default=None, help="Override diffusion beta_end")
    parser.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps")
    parser.add_argument("--diff_model", type=str, default=None, choices=["tiny", "big"], help="Override diffusion backbone")
    

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


def build_model(checkpoint: Dict, device: torch.device) -> TinyUNetDiffusion:
    """Instantiate the model and load weights from ``checkpoint``."""
    ckpt_args = checkpoint.get("args")
    base = extract_base_from_args(ckpt_args, default=10)
    diff_model = extract_arg_from_checkpoint(ckpt_args, "diff_model", "tiny")
    
    img_size = int(extract_arg_from_checkpoint(ckpt_args, "img_size", 32))
    model_state = checkpoint.get("model")
    if model_state is None:
        raise KeyError("Checkpoint is missing the 'model' state dictionary.")
    if diff_model == "big":
        
        model = BigUNetDiffusion(in_channels=3, out_channels=3, n_feat=base, height=img_size)
    else:
        model = TinyUNetDiffusion(in_ch=3, out_ch=3, base=base)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    return model


def extract_arg_from_checkpoint(args_obj: object, key: str, default):
    """Read ``key`` from checkpoint args dict/object, fallback to ``default``."""
    if args_obj is None:
        return default
    if isinstance(args_obj, dict):
        return args_obj.get(key, default)
    if hasattr(args_obj, key):
        val = getattr(args_obj, key)
        return default if val is None else val
    return default


def build_diffusion_schedule(
    timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    b_t = (beta_end - beta_start) * torch.linspace(0, 1, timesteps + 1, device=device) + beta_start
    a_t = 1.0 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1.0
    return b_t, a_t, ab_t


def perturb_input(x: Tensor, t: Tensor, noise: Tensor, ab_t: Tensor) -> Tensor:
    alpha_bar_t = ab_t[t].view(-1, 1, 1, 1)
    return alpha_bar_t.sqrt() * x + (1 - alpha_bar_t).sqrt() * noise


def denoise_add_noise(
    x: Tensor,
    t: Tensor,
    pred_noise: Tensor,
    b_t: Tensor,
    a_t: Tensor,
    ab_t: Tensor,
    z: Optional[Tensor] = None,
) -> Tensor:
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t].view(-1, 1, 1, 1) * z
    mean = (
        x - pred_noise * ((1 - a_t[t]).view(-1, 1, 1, 1) / (1 - ab_t[t]).sqrt().view(-1, 1, 1, 1))
    ) / a_t[t].sqrt().view(-1, 1, 1, 1)
    
    return mean + noise


def sample_inpaint_batch(
    model: TinyUNetDiffusion,
    ref_batch: Tensor,
    mask_known: Tensor,
    timesteps: int,
    b_t: Tensor,
    a_t: Tensor,
    ab_t: Tensor,
) -> Tensor:
    """Reverse diffuse and enforce known pixels at every step (1=known, 0=missing)."""
    batch_size = ref_batch.shape[0]
    device = ref_batch.device
    dtype = ref_batch.dtype
    inv_mask = 1.0 - mask_known

    xt = torch.randn_like(ref_batch)
    #iterative reverse process x_T -> x_0
    step_sequence = list(range(timesteps, 0, -1))

    for step in step_sequence:
        t_norm = torch.full((batch_size, 1), step / timesteps, device=device, dtype=dtype)
        pred_noise = model(xt, t_norm)

        t_step = torch.full((batch_size,), step, device=device, dtype=torch.long)
        
        z = torch.randn_like(xt) if step > 1 else torch.zeros_like(xt)
        xt = denoise_add_noise(xt, t_step, pred_noise, b_t, a_t, ab_t, z=z)
        known_t = step - 1
        known_t_batch = torch.full((batch_size,), known_t, device=device, dtype=torch.long)
        
        known_noise = torch.randn_like(ref_batch) if known_t > 0 else torch.zeros_like(ref_batch)
        x_known = perturb_input(ref_batch, known_t_batch, known_noise, ab_t)
        xt = xt * inv_mask + x_known * mask_known

    return xt.clamp(-1.0, 1.0)


def get_dataloader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    """Load the corresponding benchmark dataset"""
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
        # Only use the test set - since it is the eval_unet script
        _, dataset = random_split(train_set, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError(f"Unknown benchmark dataset: {args.benchmark}")
    
    # Print message to be sure the right benchmark and image size is used
    print(f"[Eval Data] benchmark={args.benchmark}, img_size={args.img_size}")
    print(f"[Eval Data] dataset size={len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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


def evaluate(
    model: TinyUNetDiffusion,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    timesteps: int,
    b_t: Tensor,
    a_t: Tensor,
    ab_t: Tensor,
) -> Dict[str, float]:
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
            
            preds = sample_inpaint_batch(
                model, imgs, mask_known, timesteps, b_t, a_t, ab_t,
            )
            inputs = imgs * mask_known
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
    if args.diff_model is not None:
        
        ckpt_args = checkpoint.get("args")
        if ckpt_args is None:
            checkpoint["args"] = {"diff_model": args.diff_model}
        elif isinstance(ckpt_args, dict):
            ckpt_args["diff_model"] = args.diff_model
        else:
            setattr(ckpt_args, "diff_model", args.diff_model)
    


    model = build_model(checkpoint, device)
    params = param_count(model)
    ckpt_args = checkpoint.get("args")

    timesteps = int(args.timesteps if args.timesteps is not None else extract_arg_from_checkpoint(ckpt_args, "timesteps", 500))
    beta_start = float(args.beta_start if args.beta_start is not None else extract_arg_from_checkpoint(ckpt_args, "beta_start", 1e-4))
    beta_end = float(args.beta_end if args.beta_end is not None else extract_arg_from_checkpoint(ckpt_args, "beta_end", 2e-2))
    
    
    b_t, a_t, ab_t = build_diffusion_schedule(timesteps, beta_start, beta_end, device)

    loader = get_dataloader(args, device)
    os.makedirs(args.save_dir, exist_ok=True)

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        save_dir=args.save_dir,
        timesteps=timesteps,
        b_t=b_t,
        a_t=a_t,
        ab_t=ab_t,
    )
    metrics["params"] = params
    metrics["seed"] = args.seed
    metrics["timesteps"] = timesteps
    metrics["beta_start"] = beta_start
    metrics["beta_end"] = beta_end

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    print(f"Parameters: {params:,}")
    print(f"Seed: {args.seed}")
    print(f"Timesteps: {timesteps}")
    print(f"Beta start/end: {beta_start} / {beta_end}")
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
