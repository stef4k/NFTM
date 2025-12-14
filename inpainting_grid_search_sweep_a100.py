import argparse
import os
import sys
import wandb

from image_inpainting import main

# -------------------------
# Torch cache setup
# -------------------------
torch_cache = os.path.join(os.environ.get("WORKDIR", os.getcwd()), ".cache", "torch")
os.environ["TORCH_HOME"] = torch_cache
os.makedirs(os.path.join(torch_cache, "hub", "checkpoints"), exist_ok=True)
print(f"[setup] Using cached Torch models from: {os.environ['TORCH_HOME']}")

# -------------------------
# Fixed configuration shared across sweeps
# -------------------------
fixed_config = {
    "img_size": 32,
    "benchmark": "cifar",
    "train_dataset": "cifar",
}

# -------------------------
# Sweep config builder
# -------------------------
def build_sweep_config(controller):
    """
    controller: 'dense' or 'unet'
    """

    epochs = 30 if controller == "dense" else 50

    return {
        "method": "bayes",
        "metric": {"name": "eval/psnr_final", "goal": "maximize"},
        "parameters": {
            # fixed or bounded
            "epochs": {"value": epochs},
            "K_train": {"min": 15, "max": 25, "distribution": "int_uniform"},
            "K_eval": {"min": 15, "max": 25, "distribution": "int_uniform"},

            # new hyperparameters
            "beta_start": {"min": 0.35, "max": 0.60},
            "beta_max": {"min": 0.70, "max": 1.00},
            "beta_eval_bonus": {"min": 0.07, "max": 0.15},
            "corr_clip": {"min": 0.20, "max": 0.40},
            "tv_weight": {"min": 0.0015, "max": 0.005},
            "contract_w": {"min": 5e-5, "max": 5e-4},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 100},
    }

# -------------------------
# Run-name builder
# -------------------------
def build_run_name(config, prefix):
    return (
        f"{prefix}_"
        f"ep{config['epochs']}_"
        f"ktr{config['K_train']}_"
        f"kev{config['K_eval']}_"
        f"{config['train_dataset']}_"
        f"img{config['img_size']}"
    )

# -------------------------
# The actual training run
# -------------------------
def _run_inpainting(agent_args):
    with wandb.init(
        project=agent_args.project,
        config={**fixed_config},
        name=None,
        group=agent_args.run_group,
        reinit=True,
    ):
        cfg = dict(wandb.config)
        config = {
            **fixed_config,
            **cfg,
            "controller": agent_args.controller,
        }

        run_name = build_run_name(config, agent_args.name_prefix)
        wandb.run.name = run_name

        save_dir = os.path.join(agent_args.save_root, run_name)
        os.makedirs(save_dir, exist_ok=True)

        # ----------------------------------------------
        # Construct CLI args for image_inpainting.py
        # ----------------------------------------------
        sys.argv = [
            "image_inpainting.py",
            "--img_size", str(config["img_size"]),
            "--train_dataset", str(config["train_dataset"]),
            "--benchmark", str(config["benchmark"]),

            "--epochs", str(config["epochs"]),
            "--K_train", str(config["K_train"]),
            "--K_eval", str(config["K_eval"]),
            "--controller", str(config["controller"]),

            # NEW sweep parameters
            "--beta_start", str(config["beta_start"]),
            "--beta_max", str(config["beta_max"]),
            "--beta_eval_bonus", str(config["beta_eval_bonus"]),
            "--corr_clip", str(config["corr_clip"]),
            "--tv_weight", str(config["tv_weight"]),
            "--contract_w", str(config["contract_w"]),

            # Additional fixed flags requested
            "--guard_in_train",
            "--save_metrics",
            "--use_wandb",

            "--save_dir", save_dir,
        ]

        if agent_args.agent_name:
            wandb.log({"agent": agent_args.agent_name})

        main()

# -------------------------
# Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep-id", type=str)
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument("--project", type=str, default="image_inpainting_sweep_updated")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--run-group", type=str, default=None)
    parser.add_argument("--name-prefix", type=str, default="run")
    parser.add_argument("--save-root", type=str, default="out")
    parser.add_argument("--agent-name", type=str, default=None)

    # NEW: dense OR unet
    parser.add_argument("--controller", type=str, choices=["dense", "unet"], required=True)

    return parser.parse_args()

# -------------------------
# Main launcher
# -------------------------
def main_entry():
    args = parse_args()

    # Build correct sweep config based on controller
    sweep_config = build_sweep_config(args.controller)

    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project)
        print(sweep_id, flush=True)
        if args.create_only:
            return
    else:
        sweep_id = args.sweep_id
        print(f"[sweep] Reusing existing sweep: {sweep_id}")

    # Sequential sweep agent (count determines number of configs)
    wandb.agent(
        sweep_id,
        function=lambda: _run_inpainting(args),
        project=args.project,
        count=args.count,
    )

if __name__ == "__main__":
    main_entry()
