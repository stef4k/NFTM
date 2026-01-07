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

from nftm_inpaint.metrics import lpips_dist as _metric_lpips
from nftm_inpaint.metrics import psnr as _metric_psnr
from nftm_inpaint.metrics import ssim as _metric_ssim
from nftm_inpaint.metrics import (fid_init, fid_update, fid_compute,
                     kid_init, kid_update, kid_compute)
from nftm_inpaint.unet_model import TinyUNet
import wandb

# bring in our split pieces
from nftm_inpaint.data_and_viz import set_seed, get_transform, ensure_dir, plot_metric_curve
from nftm_inpaint.rollout import parse_pyramid_arg, split_steps_eval, count_params
from nftm_inpaint.controller import TinyController, UNetController
from nftm_inpaint.engine import train_epoch, eval_steps, evaluate_metrics_full

root_dir = os.path.dirname(os.path.abspath(__file__))   # directory containing the script
benchmarks_dir = os.path.join(root_dir, "benchmarks")


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
    parser.add_argument("--img_size", type=int, default=32, choices=[32, 64, 128], help="Input image size (resize if necessary)")

    parser.add_argument("--save_metrics", action="store_true", help="save metrics.json + psnr_curve.npy to save_dir")
    parser.add_argument("--use_wandb", action="store_true", help="enable logging to Weights & Biases (wandb)")
    parser.add_argument("--pyramid", type=str, default="", help="comma-separated sizes for coarse->fine (e.g., '16,32' or '16,32,64'). Empty = single-scale.")
    parser.add_argument("--pyr_steps", type=str, default="", help="comma-separated rollout steps per level summing to K_eval (e.g., '3,9'). Empty = auto split.")
    parser.add_argument("--viz_scale", type=float, default=1.0, help="Visualization upsample scale for PNG/GIF (1.0 = native, 2.0 = 2×).")
    parser.add_argument("--step_loss", type=str, default="final", choices=["final", "linear"], help="How to accumulate data loss over rollout steps: ""'final' = only final output, 'linear' = linearly weighted per-step losses.")
    parser.add_argument("--eval_noise_sweep", action="store_true",
                    help="Run eval over multiple corruption noise types and save per-noise visuals + metrics.")

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
        train_set = ImageFolder(root=os.path.join(benchmarks_dir, "CelebAHQ"), transform=transform_train)
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
        test_set = ImageFolder(root=os.path.join(benchmarks_dir, "CelebAHQ"), transform=transform_test)
    else:
        raise ValueError(f"Unknown benchmark dataset: {args.benchmark}")
    print(f"[Data] train_dataset={train_dataset_name}, benchmark={benchmark}, img_size={img_size}")
    print(f"[Data] train_set size={len(train_set)}, test_set size={len(test_set)}")

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

    log_path = os.path.join(args.save_dir, "train_log.txt")
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered
    print(f"[logging] writing epoch logs to {log_path}", flush=True)

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
            pyramid_sizes=pyr_sizes,
            step_loss_mode=args.step_loss
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
        log_f.write(msg + "\n") # save to file

        if args.guard_in_train:
            extra = (f"         accepted steps: {stats['accepted']} | backtracks (approx): {stats['backtracks']}")
            print(extra)
            log_f.write(extra + "\n")

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
    

        # ---------------- Noise generalization sweep (EVAL ONLY) ----------------
    if args.eval_noise_sweep:
        sweep = [
            ("gaussian",     {"noise_std": args.noise_std}),
            ("uniform",      {}),
            ("saltpepper",   {"prob": 0.10}),
            ("saltpepper",   {"prob": 0.20}),
            ("poisson",      {"peak": 30.0}),
            ("poisson",      {"peak": 10.0}),
            ("speckle",      {"noise_std": 0.20}),
        ]

        sweep_dir = os.path.join(args.save_dir, "noise_sweep")
        ensure_dir(sweep_dir)

        beta_eval_sweep = beta_eval_final  # reuse final beta
        all_results = {}

        for noise_kind, kw in sweep:
            tag_parts = [noise_kind]
            for k, v in kw.items():
                tag_parts.append(f"{k}{v}")
            tag = "_".join(tag_parts)

            out_dir = os.path.join(sweep_dir, tag)
            ensure_dir(out_dir)

            # allow per-noise override of noise_std
            noise_std_use = float(kw.get("noise_std", args.noise_std))
            noise_kwargs = dict(kw)
            noise_kwargs.pop("noise_std", None)

            # Save progression grid/GIF for first batch
            _ = eval_steps(
                controller, test_loader, device,
                K_eval=args.K_eval, beta=beta_eval_sweep,
                p_missing=(args.pmin, args.pmax), block_prob=args.block_prob,
                noise_std=noise_std_use, corr_clip=args.corr_clip,
                descent_guard=False, tvw=0.0,
                save_per_epoch_dir=out_dir,
                epoch_tag=f"noise_{tag}",
                pyramid_sizes=pyr_sizes, steps_split=pyr_steps_eval,
                viz_scale=max(1.0, float(args.viz_scale)),
                noise_kind=noise_kind, noise_kwargs=noise_kwargs,
            )

            # Save full metrics
            metrics_full = evaluate_metrics_full(
                controller,
                test_loader,
                device,
                K_eval=args.K_eval,
                beta=beta_eval_sweep,
                p_missing=(args.pmin, args.pmax),
                block_prob=args.block_prob,
                noise_std=noise_std_use,
                corr_clip=args.corr_clip,
                descent_guard=False,
                tvw=0.0,
                benchmark=benchmark,
                pyramid_sizes=pyr_sizes,
                steps_split=pyr_steps_eval,
                noise_kind=noise_kind,
                noise_kwargs=noise_kwargs,
            )

            all_results[tag] = metrics_full
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics_full, f, indent=2)

        with open(os.path.join(sweep_dir, "noise_sweep.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        print("[noise-sweep] saved per-noise visuals + metrics under:", sweep_dir)
        
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
    
    log_f.close()

if __name__ == "__main__":
    main()
