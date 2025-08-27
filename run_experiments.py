#!/usr/bin/env python3
# run_experiments.py
#
# Driver for NFTM inpainting ablations:
# - heteroscedastic vs homoscedastic loss
# - rollout length / beta schedule / clipping
# - saves per-run metrics and a comparison plot

import os, sys, json, time, argparse, subprocess, csv
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_one(config, script="image_inpainting.py"):
    """Launch one run of image_inpainting.py with given config dict."""
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    cmd = [sys.executable, script]

    # args must match exactly the argparse names in image_inpainting.py
    for k in [
        "epochs","batch_size","K_train","K_eval",
        "beta_start","beta_max","beta_anneal","beta_eval_bonus",
        "corr_clip","tv_weight","pmin","pmax","block_prob",
        "noise_std","width"
    ]:
        if k in config:
            cmd += [f"--{k}", str(config[k])]

    # always save metrics
    cmd.append("--save_metrics")

    # loss selection
    loss_mode = config.get("loss", "hetero")
    cmd += ["--loss", loss_mode]
    if loss_mode == "homo":
        cmd += ["--homo_sigma", str(config.get("homo_sigma", 0.1))]

    # toggles
    cmd += ["--save_dir", save_dir]
    if config.get("save_epoch_progress", False):
        cmd.append("--save_epoch_progress")
    if config.get("guard_in_train", False):
        cmd.append("--guard_in_train")

    print("\n======================================================")
    print("Running:", config["name"])
    print("Cmd: ", " ".join(cmd))
    print("Save dir:", save_dir)
    print("======================================================\n")

    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = time.time() - start

    log_path = os.path.join(save_dir, "run.log")
    with open(log_path, "w") as f:
        f.write(proc.stdout)
    print(proc.stdout)

    metrics_path = os.path.join(save_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[WARN] metrics.json not found in {save_dir} — run likely failed.")
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    metrics["name"] = config["name"]
    metrics["duration_sec"] = round(duration, 2)
    return metrics


def plot_curves(all_metrics, out_png):
    plt.figure(figsize=(7,5))
    for m in all_metrics:
        curve = m["curve"]
        label = f"{m['name']} (final {curve[-1]:.2f} dB)"
        plt.plot(range(1, len(curve)+1), curve, marker='o', linewidth=1.5, label=label)
    plt.xlabel("NFTM step")
    plt.ylabel("PSNR (dB)")
    plt.title("Step-wise PSNR — Ablations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[plot] saved comparison curves → {out_png}")
    plt.close()

def save_csv(all_metrics, out_csv):
    ensure_dir(os.path.dirname(out_csv) or ".")
    keys = ["name","loss_mode","homo_sigma","epochs","K_train","K_eval",
            "beta_start","beta_max","beta_anneal","beta_eval_bonus",
            "corr_clip","tv_weight","final_psnr","duration_sec"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["curve"])
        for m in all_metrics:
            row = [m.get(k, "") for k in keys]
            row.append(" ".join(f"{x:.3f}" for x in m["curve"]))
            w.writerow(row)
    print(f"[csv] saved summary → {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, default="image_inpainting.py",
                    help="path to image_inpainting.py")
    ap.add_argument("--runs_dir", type=str, default="runs",
                    help="where to save experiment subfolders")
    ap.add_argument("--epochs", type=int, default=10, help="epochs per run")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    ensure_dir(runs_dir)

    # -------- Base config (does NOT fix values that we'll override frequently) --------
    base = dict(
        epochs=args.epochs,
        batch_size=128,
        K_train=8,             # curriculum still handled inside script
        K_eval=12,
        beta_start=0.28, beta_max=0.6, beta_anneal=0.03, beta_eval_bonus=0.05,
        corr_clip=0.10, tv_weight=0.01,
        pmin=0.25, pmax=0.50, block_prob=0.5,
        noise_std=0.3,
        width=48,
        save_epoch_progress=False,
        guard_in_train=False
    )

    # -------- Experiments (override AFTER unpacking base) --------
    experiments = [
        {**base, "name":"hetero_baseline", "loss":"hetero"},
        {**base, "name":"hetero_Keval20", "loss":"hetero", "K_eval":20},
        {**base, "name":"hetero_clip0p08", "loss":"hetero", "corr_clip":0.08},
        {**base, "name":"hetero_slow_beta", "loss":"hetero", "beta_start":0.20, "beta_anneal":0.02},
        {**base, "name":"homo_sigma0p10", "loss":"homo", "homo_sigma":0.10},
        {**base, "name":"homo_sigma0p10_Keval20", "loss":"homo", "homo_sigma":0.10, "K_eval":20},
    ]

    # place each run in its own folder
    for e in experiments:
        e["save_dir"] = os.path.join(runs_dir, e["name"])

    # -------- Run all --------
    results = []
    for cfg in experiments:
        m = run_one(cfg, script=args.script)
        if m is not None:
            results.append(m)

    # -------- Aggregate --------
    if results:
        plot_curves(results, os.path.join(runs_dir, "compare_curves.png"))
        save_csv(results, os.path.join(runs_dir, "summary.csv"))
        print("\n== Summary (final PSNR) ==")
        for m in sorted(results, key=lambda x: x["final_psnr"], reverse=True):
            print(f"{m['name']:<28} final PSNR: {m['final_psnr']:.2f} dB  "
                  f"[loss={m['loss_mode']}, Keval={m['K_eval']}]")
    else:
        print("No successful runs to summarize.")

if __name__ == "__main__":
    main()
