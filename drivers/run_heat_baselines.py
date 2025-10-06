"""Driver script to train and evaluate heat-equation baselines."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _format_float(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "-"


def _format_params(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _stage_header(name: str, cmd: Iterable[str]) -> None:
    joined = " ".join(cmd)
    print(f"\n[stage] {name}")
    print(f">>> {joined}")


def _stage_footer(name: str, elapsed: float) -> None:
    print(f"[stage] {name} completed in {elapsed:.1f}s")


def _run_stage(name: str, cmd: List[str], env: Mapping[str, str]) -> float:
    _stage_header(name, cmd)
    start = time.perf_counter()
    subprocess.run(cmd, check=True, env=dict(env))
    elapsed = time.perf_counter() - start
    _stage_footer(name, elapsed)
    return elapsed


def _load_json(path: Path) -> MutableMapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _print_table(summary: Mapping[str, Mapping[str, object]]) -> None:
    if not summary:
        return
    columns = [
        ("psnr", "PSNR(dB)", lambda m: _format_float(m.get("psnr"), 2)),
        ("ssim", "SSIM", lambda m: _format_float(m.get("ssim"), 3)),
        (
            "params",
            "Params",
            lambda m: _format_params(m.get("params")),
        ),
        (
            "alpha",
            "alpha",
            lambda m: _format_float(
                next(
                    (m.get(key) for key in ("alpha_hat", "alpha_estimate", "alpha") if m.get(key) is not None),
                    None,
                ),
                4,
            ),
        ),
        (
            "train_time_s",
            "Train(s)",
            lambda m: _format_float(m.get("train_time_s"), 2),
        ),
        (
            "inf_time_ms",
            "Infer(ms)",
            lambda m: _format_float(m.get("inf_time_ms"), 2),
        ),
    ]
    header = ["method", *[title for _, title, _ in columns]]
    rows: List[List[str]] = []
    for method, metrics in summary.items():
        row = [method]
        for _, _, formatter in columns:
            row.append(formatter(metrics))
        rows.append(row)
    widths = [len(col) for col in header]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    divider = "-+-".join("-" * width for width in widths)
    print("\nRESULTS")
    print(" | ".join(header[idx].ljust(widths[idx]) for idx in range(len(header))))
    print(divider)
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))


# -----------------------------------------------------------------------------
# Baseline orchestration
# -----------------------------------------------------------------------------


@dataclass
class DriverArgs:
    size: int
    timesteps: int
    seed: int
    resolved_device: str
    py_executable: str
    env: Mapping[str, str]
    cnn_train_steps: Optional[int]
    cnn_lr: Optional[float]
    cnn_batch_size: Optional[int]
    cnn_eval_batch: Optional[int]
    pinn_alpha_epochs: Optional[int]
    pinn_alpha_lr: Optional[float]
    pinn_alpha_batch_size: Optional[int]
    pinn_alpha_spatial: bool
    eval_batch: Optional[int]
    include_pinn_mlp: bool
    pinn_mlp_epochs: Optional[int]
    pinn_mlp_lr: Optional[float]
    pinn_mlp_hidden: Optional[int]
    pinn_mlp_layers: Optional[int]
    pinn_mlp_fourier: Optional[int]
    pinn_mlp_no_cuda: bool
    pinn_mlp_out_root: str


def _run_cnn(cfg: DriverArgs) -> Mapping[str, object]:
    metrics_path = Path("out") / f"cnn_{cfg.size}" / "metrics.json"
    ckpt_path = Path("out") / f"cnn_{cfg.size}" / "ckpt.pt"

    train_cmd = [
        cfg.py_executable,
        "-m",
        "baselines.cnn_baseline",
        "--size",
        str(cfg.size),
        "--timesteps",
        str(cfg.timesteps),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.resolved_device,
    ]
    if cfg.cnn_train_steps is not None:
        train_cmd += ["--train_steps", str(cfg.cnn_train_steps)]
    if cfg.cnn_lr is not None:
        train_cmd += ["--lr", str(cfg.cnn_lr)]
    if cfg.cnn_batch_size is not None:
        train_cmd += ["--batch_size", str(cfg.cnn_batch_size)]
    train_elapsed = _run_stage("cnn_train", train_cmd, cfg.env)

    eval_cmd = [
        cfg.py_executable,
        "-m",
        "eval.eval_cnn",
        "--ckpt",
        str(ckpt_path),
        "--size",
        str(cfg.size),
        "--timesteps",
        str(cfg.timesteps),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.resolved_device,
    ]
    eval_batch = cfg.cnn_eval_batch or cfg.eval_batch
    if eval_batch is not None:
        eval_cmd += ["--batch", str(eval_batch)]
    _run_stage("cnn_eval", eval_cmd, cfg.env)

    metrics = _load_json(metrics_path)
    metrics["train_time_s"] = train_elapsed
    return metrics


def _run_pinn_alpha(cfg: DriverArgs) -> Mapping[str, object]:
    metrics_path = Path("out") / f"pinn_alpha_{cfg.size}" / "metrics.json"
    ckpt_path = Path("out") / f"pinn_alpha_{cfg.size}" / "ckpt.pt"

    train_cmd = [
        cfg.py_executable,
        "-m",
        "baselines.pinn_alpha",
        "--size",
        str(cfg.size),
        "--timesteps",
        str(cfg.timesteps),
        "--seed",
        str(cfg.seed),
    ]
    if cfg.pinn_alpha_spatial:
        train_cmd.append("--spatial_alpha")
    if cfg.pinn_alpha_epochs is not None:
        train_cmd += ["--epochs", str(cfg.pinn_alpha_epochs)]
    if cfg.pinn_alpha_lr is not None:
        train_cmd += ["--lr", str(cfg.pinn_alpha_lr)]
    if cfg.pinn_alpha_batch_size is not None:
        train_cmd += ["--batch_size", str(cfg.pinn_alpha_batch_size)]
    train_elapsed = _run_stage("pinn_alpha_train", train_cmd, cfg.env)

    eval_cmd = [
        cfg.py_executable,
        "-m",
        "eval.eval_pinn_alpha",
        "--ckpt",
        str(ckpt_path),
        "--size",
        str(cfg.size),
        "--timesteps",
        str(cfg.timesteps),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.resolved_device,
    ]
    eval_batch = cfg.eval_batch
    if eval_batch is not None:
        eval_cmd += ["--batch", str(eval_batch)]
    _run_stage("pinn_alpha_eval", eval_cmd, cfg.env)

    metrics = _load_json(metrics_path)
    metrics["train_time_s"] = train_elapsed
    return metrics


def _run_pinn_mlp(cfg: DriverArgs) -> Mapping[str, object]:
    metrics_path = Path(cfg.pinn_mlp_out_root) / f"pinn_mlp_{cfg.size}" / "metrics.json"

    train_cmd = [
        cfg.py_executable,
        "-m",
        "baselines.pinn_mlp",
        "--size",
        str(cfg.size),
        "--timesteps",
        str(cfg.timesteps),
        "--seed",
        str(cfg.seed),
        "--out_dir",
        cfg.pinn_mlp_out_root,
    ]
    if cfg.pinn_mlp_epochs is not None:
        train_cmd += ["--epochs", str(cfg.pinn_mlp_epochs)]
    if cfg.pinn_mlp_lr is not None:
        train_cmd += ["--lr", str(cfg.pinn_mlp_lr)]
    if cfg.pinn_mlp_hidden is not None:
        train_cmd += ["--hidden", str(cfg.pinn_mlp_hidden)]
    if cfg.pinn_mlp_layers is not None:
        train_cmd += ["--layers", str(cfg.pinn_mlp_layers)]
    if cfg.pinn_mlp_fourier is not None:
        train_cmd += ["--fourier", str(cfg.pinn_mlp_fourier)]
    if cfg.pinn_mlp_no_cuda or cfg.resolved_device == "cpu":
        train_cmd.append("--no_cuda")
    train_elapsed = _run_stage("pinn_mlp_train", train_cmd, cfg.env)

    metrics = _load_json(metrics_path)
    if "train_time" in metrics:
        metrics["train_time_s"] = metrics.get("train_time")
    else:
        metrics["train_time_s"] = train_elapsed
    metrics.setdefault("ssim", None)
    metrics.setdefault("inf_time_ms", None)
    return metrics


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32, help="Spatial resolution")
    parser.add_argument("--timesteps", type=int, default=10, help="Rollout length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device",
    )
    parser.add_argument("--cnn-train-steps", type=int, dest="cnn_train_steps")
    parser.add_argument("--cnn-lr", type=float, dest="cnn_lr")
    parser.add_argument("--cnn-batch-size", type=int, dest="cnn_batch_size")
    parser.add_argument("--cnn-eval-batch", type=int, dest="cnn_eval_batch")
    parser.add_argument("--pinn-alpha-epochs", type=int, dest="pinn_alpha_epochs")
    parser.add_argument("--pinn-alpha-lr", type=float, dest="pinn_alpha_lr")
    parser.add_argument(
        "--pinn-alpha-batch-size", type=int, dest="pinn_alpha_batch_size"
    )
    parser.add_argument(
        "--pinn-alpha-spatial",
        action="store_true",
        help="Train the spatial α variant",
    )
    parser.add_argument("--eval-batch", type=int, dest="eval_batch")
    parser.add_argument(
        "--include-pinn-mlp",
        action="store_true",
        help="Include the PINN-MLP baseline",
    )
    parser.add_argument("--pinn-mlp-epochs", type=int, dest="pinn_mlp_epochs")
    parser.add_argument("--pinn-mlp-lr", type=float, dest="pinn_mlp_lr")
    parser.add_argument("--pinn-mlp-hidden", type=int, dest="pinn_mlp_hidden")
    parser.add_argument("--pinn-mlp-layers", type=int, dest="pinn_mlp_layers")
    parser.add_argument("--pinn-mlp-fourier", type=int, dest="pinn_mlp_fourier")
    parser.add_argument(
        "--pinn-mlp-no-cuda",
        action="store_true",
        help="Disable CUDA for PINN-MLP regardless of availability",
    )
    args = parser.parse_args(argv)

    requested_device, resolved_device = _resolve_device(args.device)
    py_executable = sys.executable

    base_env = os.environ.copy()
    if resolved_device == "cpu" and requested_device != "auto":
        base_env["CUDA_VISIBLE_DEVICES"] = "-1"
    cfg = DriverArgs(
        size=args.size,
        timesteps=args.timesteps,
        seed=args.seed,
        resolved_device=resolved_device,
        py_executable=py_executable,
        env=base_env,
        cnn_train_steps=args.cnn_train_steps,
        cnn_lr=args.cnn_lr,
        cnn_batch_size=args.cnn_batch_size,
        cnn_eval_batch=args.cnn_eval_batch,
        pinn_alpha_epochs=args.pinn_alpha_epochs,
        pinn_alpha_lr=args.pinn_alpha_lr,
        pinn_alpha_batch_size=args.pinn_alpha_batch_size,
        pinn_alpha_spatial=args.pinn_alpha_spatial,
        eval_batch=args.eval_batch,
        include_pinn_mlp=args.include_pinn_mlp,
        pinn_mlp_epochs=args.pinn_mlp_epochs,
        pinn_mlp_lr=args.pinn_mlp_lr,
        pinn_mlp_hidden=args.pinn_mlp_hidden,
        pinn_mlp_layers=args.pinn_mlp_layers,
        pinn_mlp_fourier=args.pinn_mlp_fourier,
        pinn_mlp_no_cuda=args.pinn_mlp_no_cuda,
        pinn_mlp_out_root="out",
    )

    summary: Dict[str, Mapping[str, object]] = {}

    summary["cnn"] = _run_cnn(cfg)
    summary["pinn_alpha"] = _run_pinn_alpha(cfg)

    if cfg.include_pinn_mlp:
        summary["pinn_mlp"] = _run_pinn_mlp(cfg)

    summary_path = Path("out") / f"summary_{cfg.size}.json"
    _ensure_dir(summary_path.parent)
    payload = {
        "size": cfg.size,
        "timesteps": cfg.timesteps,
        "seed": cfg.seed,
        "device": resolved_device,
        "methods": summary,
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nWrote summary to {summary_path}")

    _print_table(summary)


if __name__ == "__main__":
    main()
