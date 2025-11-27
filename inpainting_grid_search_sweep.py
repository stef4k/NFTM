import argparse
import os
import sys
import wandb

from image_inpainting import main

# Use shared Torch cache
torch_cache = os.path.join(os.environ.get("WORKDIR", os.getcwd()), ".cache", "torch")
os.environ["TORCH_HOME"] = torch_cache
os.makedirs(os.path.join(torch_cache, "hub", "checkpoints"), exist_ok=True)
print(f"[setup] Using cached Torch models from: {os.environ['TORCH_HOME']}")

fixed_config = {
    "controller": "dense",
    "img_size": 64,
    "benchmark": "celebahq",
    "train_dataset": "celebahq",
}

def build_sweep_config():
    return {
        "method": "bayes",
        "metric": {"name": "eval/psnr_final", "goal": "maximize"},
        "parameters": {
            "epochs": {"min": 5, "max": 50, "distribution": "int_uniform"},
            "K_train": {"min": 1, "max": 50, "distribution": "int_uniform"},
            "K_eval": {"min": 1, "max": 100, "distribution": "int_uniform"},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 100},
    }

def build_run_name(config, prefix):
    return (
        f"{prefix}_"
        f"ep{config['epochs']}_"
        f"ktr{config['K_train']}_"
        f"kev{config['K_eval']}_"
        f"{config['train_dataset']}_"
        f"img{config['img_size']}"
    )

def _run_inpainting(agent_args):
    with wandb.init(
        project=agent_args.project,
        config={**fixed_config},
        name=None,
        group=agent_args.run_group,
        reinit=True,
    ):

        cfg = dict(wandb.config)
        config = {**fixed_config, **cfg}

        run_name = build_run_name(config, agent_args.name_prefix)
        wandb.run.name = run_name

        save_dir = os.path.join(agent_args.save_root, run_name)
        os.makedirs(save_dir, exist_ok=True)

        sys.argv = [
            "image_inpainting.py",
            "--img_size", str(config["img_size"]),
            "--train_dataset", str(config["train_dataset"]),
            "--benchmark", str(config["benchmark"]),
            "--epochs", str(config["epochs"]),
            "--K_train", str(config["K_train"]),
            "--K_eval", str(config["K_eval"]),
            "--controller", config["controller"],
            "--save_metrics",
            "--use_wandb",
            "--save_dir", save_dir,
        ]

        if agent_args.agent_name:
            wandb.log({"agent": agent_args.agent_name})

        main()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str)
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument("--project", type=str, default="image_inpainting_sweep")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--run-group", type=str, default=None)
    parser.add_argument("--name-prefix", type=str, default="bayes")
    parser.add_argument("--save-root", type=str, default="out")
    parser.add_argument("--agent-name", type=str, default=None)
    return parser.parse_args()

def main_entry():
    args = parse_args()

    sweep_id = args.sweep_id
    sweep_config = build_sweep_config()

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project)
        print(sweep_id, flush=True)
    else:
        print(f"[sweep] Reusing existing sweep: {sweep_id}")

    if args.create_only:
        return

    wandb.agent(
        sweep_id,
        function=lambda: _run_inpainting(args),
        project=args.project,
        count=args.count,
    )

if __name__ == "__main__":
    main_entry()
