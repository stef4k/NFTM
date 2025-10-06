#!/usr/bin/env python3
"""One-click orchestration for NFTM inpainting experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import time
from datetime import datetime
from typing import Dict, Iterable, List, Tuple


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_device(device: str) -> Tuple[str, str]:
    if device != "auto":
        return device, device
    resolved = "cpu"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            resolved = "cuda"
    except Exception:
        resolved = "cpu"
    return "auto", resolved


def _format_float(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def _stage_header(name: str, cmd: Iterable[str]) -> None:
    joined = " ".join(cmd)
    print(f"\n[stage] {name}")
    print(f">>> {joined}")


def _stage_footer(name: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"[stage] {name} completed in {elapsed:.1f}s")


def _run_stage(name: str, cmd: List[str]) -> None:
    _stage_header(name, cmd)
    start = time.time()
    subprocess.run(cmd, check=True)
    _stage_footer(name, start)


def _load_metrics(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _pretty_print(summary: Dict[str, Dict[str, float]], methods: Iterable[str]) -> None:
    columns = [
        "psnr_all",
        "psnr_miss",
        "ssim_all",
        "ssim_miss",
        "lpips_all",
        "lpips_miss",
        "params",
    ]
    header = ["method", *columns]
    rows: List[List[str]] = []
    for method in methods:
        if method not in summary:
            continue
        metrics = summary[method]
        row = [method]
        for column in columns:
            value = metrics.get(column)
            row.append(_format_float(value))
        rows.append(row)
    if not rows:
        return
    widths = [len(col) for col in header]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    divider = "-+-".join("-" * width for width in widths)
    print("\nRESULTS")
    print(" | ".join(title.ljust(widths[idx]) for idx, title in enumerate(header)))
    print(divider)
    for row in rows:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out_all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--unet_base", type=int, default=10)
    parser.add_argument("--target_params", type=int, default=46375)
    parser.add_argument("--tvl1_iters", type=int, default=250)
    parser.add_argument("--tvl1_lam", type=float, default=80.0)
    parser.add_argument("--tvl1_tvw", type=float, default=0.10)
    parser.add_argument("--tvl1_tau_p", type=float, default=0.25)
    parser.add_argument("--tvl1_tau_d", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--include_nftm", action="store_true")
    args = parser.parse_args(argv)

    requested_device, resolved_device = _resolve_device(args.device)
    base_dir = args.out
    py_executable = sys.executable

    dirs = {
        "unet": os.path.join(base_dir, "unet"),
        "unet_eval": os.path.join(base_dir, "unet_eval"),
        "tvl1": os.path.join(base_dir, "tvl1"),
    }
    if args.include_nftm:
        dirs["nftm"] = os.path.join(base_dir, "nftm")

    for path in [base_dir, *dirs.values()]:
        _ensure_dir(path)

    try:
        _run_stage(
            "train_unet",
            [
                py_executable,
                "train_unet.py",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.wd),
                "--tv_weight",
                str(args.tv_weight),
                "--seed",
                str(args.seed),
                "--save_dir",
                dirs["unet"],
                "--base",
                str(args.unet_base),
                "--target_params",
                str(args.target_params),
                "--num_workers",
                str(args.num_workers),
                "--device",
                resolved_device,
            ],
        )

        _run_stage(
            "eval_unet",
            [
                py_executable,
                "eval_unet.py",
                "--ckpt",
                os.path.join(dirs["unet"], "ckpt.pt"),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--save_dir",
                dirs["unet_eval"],
                "--device",
                resolved_device,
            ],
        )

        _run_stage(
            "tvl1_baseline",
            [
                py_executable,
                "-m",
                "baselines.inpainting.tvl1_baseline",
                "--iters",
                str(args.tvl1_iters),
                "--lam",
                str(args.tvl1_lam),
                "--tvw",
                str(args.tvl1_tvw),
                "--tau_p",
                str(args.tvl1_tau_p),
                "--tau_d",
                str(args.tvl1_tau_d),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--save_dir",
                dirs["tvl1"],
                "--device",
                resolved_device,
            ],
        )

        if args.include_nftm and "nftm" in dirs:
            _run_stage(
                "nftm",
                [
                    py_executable,
                    "image_inpainting.py",
                    "--save_metrics",
                    "--save_dir",
                    dirs["nftm"],
                ],
            )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure path
        cmd = " ".join(str(part) for part in exc.cmd)
        print(f"[error] command failed with exit code {exc.returncode}: {cmd}")
        raise SystemExit(exc.returncode)

    summary: Dict[str, Dict[str, float]] = {}
    try:
        summary["unet"] = _load_metrics(os.path.join(dirs["unet_eval"], "metrics.json"))
        summary["tvl1"] = _load_metrics(os.path.join(dirs["tvl1"], "metrics.json"))
        if args.include_nftm and "nftm" in dirs:
            summary["nftm"] = _load_metrics(os.path.join(dirs["nftm"], "metrics.json"))
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)

    summary["meta"] = {
        "seed": args.seed,
        "device": resolved_device,
        "requested_device": requested_device,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    summary_path = os.path.join(base_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[info] summary written to {summary_path}")

    _pretty_print(summary, ["unet", "tvl1", "nftm"])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
