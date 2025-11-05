#!/usr/bin/env python3
"""One-click orchestration for NFTM inpainting experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


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
        "kid",
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


def _resolve_mask_dir(mask_dir: str | None) -> str | None:
    if not mask_dir:
        return None
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "image_inpainting.py")
    )
    try:
        with open(script_path, "r", encoding="utf-8") as fp:
            if "--mask_dir" in fp.read():
                return mask_dir
    except OSError:
        pass
    print("[warn] --mask_dir ignored; image_inpainting.py does not support this argument")
    return None


def _run_nftm_commands(
    py_executable: str,
    controller_dirs: Dict[str, str],
    *,
    epochs: int,
    k_train: int,
    k_eval: int,
    loss: str,
    device: str,
    seed: int,
    extra_args: List[str],
    mask_dir: str | None,
) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    for controller, save_dir in controller_dirs.items():
        cmd = [
            py_executable,
            "image_inpainting.py",
            "--controller",
            controller,
            "--save_dir",
            save_dir,
            "--epochs",
            str(epochs),
            "--K_train",
            str(k_train),
            "--K_eval",
            str(k_eval),
            "--loss",
            loss,
            "--seed",
            str(seed),
            "--device",
            device,
            "--save_metrics",
        ]
        if mask_dir and "--mask_dir" not in extra_args:
            cmd.extend(["--mask_dir", mask_dir])
        if extra_args:
            cmd.extend(extra_args)
        stage_name = f"nftm_{controller}"
        _run_stage(stage_name, cmd)
        runs.append((controller, save_dir))
    return runs


def _collect_nftm_metrics(
    runs: List[Tuple[str, str]],
    *,
    loss: str,
    epochs: int,
    k_train: int,
    k_eval: int,
    device: str,
    seed: int,
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for controller, save_dir in runs:
        metrics_path = os.path.join(save_dir, "metrics.json")
        metrics = _load_metrics(metrics_path)
        row: Dict[str, Any] = {
            "controller": controller,
            "loss": metrics.get("loss_mode", loss),
            "epochs": metrics.get("epochs", epochs),
            "K_train": metrics.get("K_train", k_train),
            "K_eval": metrics.get("K_eval", k_eval),
            "device": metrics.get("device", device),
            "seed": metrics.get("seed", seed),
            "final_psnr": metrics.get("final_psnr"),
            "save_dir": save_dir,
        }
        for key in [
            "psnr_all",
            "psnr_miss",
            "ssim_all",
            "ssim_miss",
            "lpips_all",
            "lpips_miss",
            "runtime_ms_per_image",
        ]:
            row[key] = metrics.get(key)
        collected.append(row)
    return collected


def _write_nftm_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "controller",
        "loss",
        "epochs",
        "K_train",
        "K_eval",
        "device",
        "seed",
        "psnr_all",
        "psnr_miss",
        "ssim_all",
        "ssim_miss",
        "lpips_all",
        "lpips_miss",
        "runtime_ms_per_image",
        "final_psnr",
        "save_dir",
    ]
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    print(f"[info] NFTM summary written to {path}")


def _print_nftm_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "controller",
        "psnr_all",
        "psnr_miss",
        "ssim_all",
        "ssim_miss",
        "lpips_all",
        "lpips_miss",
        "final_psnr",
    ]
    widths = [len(col) for col in columns]
    table_rows: List[List[str]] = []
    for row in rows:
        formatted_row: List[str] = []
        for idx, column in enumerate(columns):
            if column == "controller":
                cell = str(row.get(column, ""))
            else:
                cell = _format_float(row.get(column))
            widths[idx] = max(widths[idx], len(cell))
            formatted_row.append(cell)
        table_rows.append(formatted_row)
    print("\nNFTM SUMMARY")
    print(" | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns))))
    print("-+-".join("-" * width for width in widths))
    for formatted_row in table_rows:
        print(" | ".join(formatted_row[idx].ljust(widths[idx]) for idx in range(len(columns))))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out_all")
    parser.add_argument("--epochs", type=int, default=None)
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
    parser.add_argument("--beta_start", type=float, default=0.28, help="initial beta (step size)")
    parser.add_argument("--beta_max", type=float, default=0.6, help="cap on beta during training")
    parser.add_argument("--beta_anneal", type=float, default=0.03, help="per-epoch beta increment")
    parser.add_argument("--beta_eval_bonus", type=float, default=0.05, help="extra beta for eval")
    parser.add_argument("--corr_clip", type=float, default=0.1, help="max per-step correction magnitude (base)")
    parser.add_argument("--pmin", type=float, default=0.25, help="min missing fraction")
    parser.add_argument("--pmax", type=float, default=0.5, help="max missing fraction")
    parser.add_argument("--block_prob", type=float, default=0.5, help="probability to add random occlusion blocks")
    parser.add_argument("--noise_std", type=float, default=0.3, help="corruption noise std for missing pixels")
    parser.add_argument("--width", type=int, default=48, help="controller width")
    parser.add_argument("--save_metrics", action="store_true", help="save metrics.json + psnr_curve.npy to save_dir")
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--include_nftm", action="store_true")
    parser.add_argument("--save_root", type=str, default="runs/inpainting")
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument("--K_train", type=int, default=None)
    parser.add_argument("--K_eval", type=int, default=None)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)

    baseline_epochs = args.epochs if args.epochs is not None else 20
    nftm_epochs = args.epochs if args.epochs is not None else 30
    nftm_k_train = args.K_train if args.K_train is not None else 20
    nftm_k_eval = args.K_eval if args.K_eval is not None else 30

    requested_device, resolved_device = _resolve_device(args.device)
    base_dir = args.out
    py_executable = sys.executable

    nftm_controller_dirs = {
        "dense": os.path.join(args.save_root, "nftm_dense"),
        "unet": os.path.join(args.save_root, "nftm_unet"),
    }
    mask_dir_value = _resolve_mask_dir(args.mask_dir if args.include_nftm else None)

    dirs = {
        "unet": os.path.join(base_dir, "unet"),
        "unet_eval": os.path.join(base_dir, "unet_eval"),
        "tvl1": os.path.join(base_dir, "tvl1"),
        "unet_recursive": os.path.join(base_dir, "unet_recursive"),
        "unet_recursive_eval": os.path.join(base_dir, "unet_recursive_eval"),
    }
    if args.include_nftm:
        dirs["nftm"] = os.path.join(base_dir, "nftm")

    for path in [base_dir, *dirs.values()]:
        _ensure_dir(path)

    nftm_runs: List[Tuple[str, str]] = []
    try:
        _run_stage(
            "train_unet",
            [
                py_executable,
                "train_unet.py",
                "--epochs",
                str(baseline_epochs),
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
            "train_unet_recur",
            [
                py_executable,
                "train_unet_recur.py",
                "--epochs",
                str(baseline_epochs),
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
                dirs["unet_recursive"],
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
            "eval_unet_recur",
            [
                py_executable,
                "eval_unet_recur.py",
                "--ckpt",
                os.path.join(dirs["unet_recursive"], "ckpt.pt"),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--save_dir",
                dirs["unet_recursive_eval"],
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
        if args.include_nftm:
            for path in [args.save_root, *nftm_controller_dirs.values()]:
                _ensure_dir(path)
            nftm_runs = _run_nftm_commands(
                py_executable,
                nftm_controller_dirs,
                epochs=nftm_epochs,
                k_train=nftm_k_train,
                k_eval=nftm_k_eval,
                loss=args.loss,
                device=resolved_device,
                seed=args.seed,
                extra_args=list(args.extra),
                mask_dir=mask_dir_value,
            )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure path
        cmd = " ".join(str(part) for part in exc.cmd)
        print(f"[error] command failed with exit code {exc.returncode}: {cmd}")
        raise SystemExit(exc.returncode)

    summary: Dict[str, Dict[str, float]] = {}
    try:
        summary["unet"] = _load_metrics(os.path.join(dirs["unet_eval"], "metrics.json"))
        summary["unet_recursive"] = _load_metrics(os.path.join(dirs["unet_recursive_eval"], "metrics.json"))
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
        "epochs": baseline_epochs,
        "batch_size": args.batch_size,
    }

    summary_path = os.path.join(base_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[info] summary written to {summary_path}")

    _pretty_print(summary, ["unet","unet_recursive", "tvl1", "nftm"])

    if args.include_nftm and nftm_runs:
        try:
            nftm_rows = _collect_nftm_metrics(
                nftm_runs,
                loss=args.loss,
                epochs=nftm_epochs,
                k_train=nftm_k_train,
                k_eval=nftm_k_eval,
                device=resolved_device,
                seed=args.seed,
            )
        except FileNotFoundError as exc:
            print(f"[error] {exc}")
            raise SystemExit(1)
        summary_csv_path = os.path.join(args.save_root, "summary.csv")
        _write_nftm_summary_csv(summary_csv_path, nftm_rows)
        _print_nftm_table(nftm_rows)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
