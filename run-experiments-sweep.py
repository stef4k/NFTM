#!/usr/bin/env python3
# run_experiments-sweep.py
#
# Sweeps NFTM inpainting settings (MSE-only after the change).
# Compares controllers, beta schedules, clipping, rollout depth, and a few toggles.
# Writes per-run metrics and an aggregate CSV + curves plot.
# ! We will need a proper GPU to test this on multiple epochs and larger images !

import os, sys, json, time, argparse, subprocess, csv
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _forwardable_keys():
    # MUST match argparse names in image_inpainting.py
    return [
        "epochs","batch_size","K_train","K_eval",
        "beta_start","beta_max","beta_anneal","beta_eval_bonus",
        "corr_clip","tv_weight","pmin","pmax","block_prob",
        "noise_std","width","controller","unet_base",
        "contract_w","seed","device","benchmark",
    ]

def run_one(config, script="image_inpainting.py"):
    """Launch one run of image_inpainting.py with given config dict."""
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    cmd = [sys.executable, script]

    # Forward args that the training script actually knows about
    for k in _forwardable_keys():
        if k in config and config[k] is not None:
            cmd += [f"--{k}", str(config[k])]

    # always save metrics
    cmd.append("--save_metrics")

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

    # persist stdout
    with open(os.path.join(save_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    print(proc.stdout)

    metrics_path = os.path.join(save_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[WARN] metrics.json not found in {save_dir} - run likely failed.")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics["name"] = config["name"]
    metrics["duration_sec"] = round(duration, 2)
    metrics["_cfg"] = {k: config.get(k, None) for k in _forwardable_keys()}
    metrics["_toggles"] = {
        "guard_in_train": bool(config.get("guard_in_train", False)),
        "save_epoch_progress": bool(config.get("save_epoch_progress", False)),
    }
    return metrics


def plot_curves(all_metrics, out_png):
    plt.figure(figsize=(8,5))
    for m in all_metrics:
        curve = m.get("curve", [])
        if not curve:
            continue
        label_bits = [m["name"]]
        if "controller" in m:
            label_bits.append(m["controller"])
        label = " | ".join(label_bits) + f"  (final {curve[-1]:.2f} dB)"
        xs = list(range(1, len(curve)+1))
        plt.plot(xs, curve, marker='o', linewidth=1.5, label=label)
    plt.xlabel("NFTM step")
    plt.ylabel("PSNR (dB)")
    plt.title("Step-wise PSNR — Ablations")
    plt.grid(True, alpha=0.3)
    # legend outside plot
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        frameon=False
    )
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[plot] saved comparison curves -> {out_png}")
    plt.close()


def save_csv(all_metrics, out_csv):
    ensure_dir(os.path.dirname(out_csv) or ".")
    keys = [
        "name","controller","unet_base","epochs","K_train","K_eval",
        "beta_start","beta_max","beta_anneal","beta_eval_bonus",
        "corr_clip","tv_weight","contract_w","pmin","pmax","block_prob",
        "noise_std","final_psnr","final_ssim","final_lpips",
        "psnr_all","psnr_miss","ssim_all","ssim_miss","lpips_all","lpips_miss",
        "fid","kid","params","seed","duration_sec",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys + ["curve"])
        for m in all_metrics:
            row = []
            for k in keys:
                # Populate from metrics JSON or its embedded cfg
                if k in m:
                    row.append(m.get(k, ""))
                else:
                    row.append(m.get("_cfg", {}).get(k, ""))
            curve = m.get("curve", [])
            row.append(" ".join(f"{x:.3f}" for x in curve) if curve else "")
            w.writerow(row)
    print(f"[csv] saved summary -> {out_csv}")


def _make_grid(runs_dir, epochs):
    """
    Build a compact but useful grid:
      - controllers: dense, unet
      - beta schedule: default vs slow
      - clipping: 0.10 vs 0.08
      - K_eval: 12 vs 20
      - optional toggles on a few configs: guard_in_train, tv_weight, contract_w
    Keep it small enough for Colab while showing which 'tricks' help.
    """
    base = dict(
        epochs=epochs,
        batch_size=128, # might need to reduce for Colab until we have a proper GPU
        K_train=8, # curriculum handled inside script
        K_eval=12,
        beta_start=0.28, beta_max=0.6, beta_anneal=0.03, beta_eval_bonus=0.05,
        corr_clip=0.10, tv_weight=0.01,
        pmin=0.25, pmax=0.50, block_prob=0.5,
        noise_std=0.3,
        width=48,
        contract_w=1e-3,
        seed=0,
        device="cuda", # defaults to cuda if available
        benchmark="cifar", # for now, might be interesting to try others once we have GPU access
        save_epoch_progress=False,
        guard_in_train=False,
    )

    controllers = ["dense", "unet"]
    schedule_variants = [
        dict(tag="sched_default", beta_start=0.28, beta_anneal=0.03),
        dict(tag="sched_slow",    beta_start=0.20, beta_anneal=0.02),
    ]
    clip_vals = [0.10, 0.08]
    kevals = [12, 20]

    runs = []

    # Core grid
    for ctrl in controllers:
        for sched in schedule_variants:
            for clip in clip_vals:
                for ke in kevals:
                    cfg = {**base}
                    cfg.update(dict(controller=ctrl,
                                    beta_start=sched["beta_start"],
                                    beta_anneal=sched["beta_anneal"],
                                    corr_clip=clip,
                                    K_eval=ke))
                    name = f"{ctrl}_{sched['tag']}_clip{str(clip).replace('.','p')}_Keval{ke}"
                    cfg["name"] = name
                    cfg["save_dir"] = os.path.join(runs_dir, name)
                    runs.append(cfg)

    # A few toggled variants on top of the default baseline for each controller
    for ctrl in controllers:
        # Guard in train (stability vs speed)
        cfg = {**base, "controller": ctrl, "guard_in_train": True}
        cfg["name"] = f"{ctrl}_guard_train"
        cfg["save_dir"] = os.path.join(runs_dir, cfg["name"])
        runs.append(cfg)

        # No TV (tv_weight=0.0)
        cfg = {**base, "controller": ctrl, "tv_weight": 0.0}
        cfg["name"] = f"{ctrl}_no_tv"
        cfg["save_dir"] = os.path.join(runs_dir, cfg["name"])
        runs.append(cfg)

        # No contractive penalty (contract_w=0.0)
        cfg = {**base, "controller": ctrl, "contract_w": 0.0}
        cfg["name"] = f"{ctrl}_no_contract"
        cfg["save_dir"] = os.path.join(runs_dir, cfg["name"])
        runs.append(cfg)

    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, default="image_inpainting.py",
                    help="path to image_inpainting.py")
    ap.add_argument("--runs_dir", type=str, default="runs/sweep",
                    help="where to save experiment subfolders")
    ap.add_argument("--epochs", type=int, default=8, help="epochs per run")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    ensure_dir(runs_dir)

    # Build experiment grid
    experiments = _make_grid(runs_dir, epochs=args.epochs)

    # -------- Run all --------
    results = []
    for cfg in experiments:
        m = run_one(cfg, script=args.script)
        if m is not None:
            # Keep a few handy top-level fields for plotting/printing
            m["controller"] = cfg.get("controller", "")
            results.append(m)

    # -------- Aggregate --------
    if results:
        plot_curves(results, os.path.join(runs_dir, "compare_curves.png"))
        save_csv(results, os.path.join(runs_dir, "summary.csv"))

        # Leaderboard by final PSNR
        winners = sorted(
            [r for r in results if r.get("final_psnr") is not None],
            key=lambda x: x["final_psnr"],
            reverse=True
        )
        print("\n== Summary (final PSNR) ==")
        for m in winners[:12]:
            ke = m.get("_cfg", {}).get("K_eval", "")
            clip = m.get("_cfg", {}).get("corr_clip", "")
            sched = f"β0={m.get('_cfg',{}).get('beta_start','?')}, dβ={m.get('_cfg',{}).get('beta_anneal','?')}"
            tvw = m.get("_cfg", {}).get("tv_weight", "")
            cw = m.get("_cfg", {}).get("contract_w", "")
            guard = m.get("_toggles", {}).get("guard_in_train", False)
            print(f"{m['name']:<36} PSNR={m['final_psnr']:.2f}  "
                  f"[ctrl={m.get('controller','')}, Keval={ke}, clip={clip}, {sched}, tv={tvw}, cw={cw}, guard={guard}]")
    else:
        print("No successful runs to summarize.")

if __name__ == "__main__":
    main()
