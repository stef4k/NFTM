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
            "epochs": {"values": [i for i in range(5, 50, 2)]},
            "K_train": {"values": [i for i in range(1, 50, 1)]},
            "K_eval": {"values": [i for i in range(1, 100, 2)]},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 100},
    }


def _run_inpainting(agent_args):
    cfg = dict(wandb.config)
    config = {**fixed_config, **cfg}
    run_name = (
        f"{agent_args.name_prefix}_ep{config['epochs']}_"
        f"ktr{config['K_train']}_kev{config['K_eval']}_"
        f"{config['train_dataset']}_img{config['img_size']}"
    )
    save_dir = os.path.join(agent_args.save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)

    with wandb.init(
        project=agent_args.project,
        config=config,
        name=run_name,
        group=agent_args.run_group,
        reinit=True,
    ):
        sys.argv = [
            "image_inpainting.py",
            "--img_size",
            str(config["img_size"]),
            "--train_dataset",
            str(config["train_dataset"]),
            "--benchmark",
            str(config["benchmark"]),
            "--epochs",
            str(config["epochs"]),
            "--K_train",
            str(config["K_train"]),
            "--K_eval",
            str(config["K_eval"]),
            "--controller",
            config["controller"],
            "--save_metrics",
            "--use_wandb",
            "--save_dir",
            save_dir,
        ]
        if agent_args.agent_name:
            wandb.log({"agent": agent_args.agent_name})
        main()


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian sweep driver for NFTM inpainting")
    parser.add_argument("--sweep-id", type=str, help="Existing W&B sweep id to attach agents")
    parser.add_argument("--create-only", action="store_true", help="Create a sweep and print id, then exit")
    parser.add_argument("--project", type=str, default="image_inpainting_sweep", help="W&B project name")
    parser.add_argument("--count", type=int, default=8, help="Number of runs for this agent")
    parser.add_argument("--run-group", type=str, default=None, help="Optional W&B run group for this sweep")
    parser.add_argument("--name-prefix", type=str, default="bayes", help="Run name prefix")
    parser.add_argument("--save-root", type=str, default="out", help="Root directory for run outputs")
    parser.add_argument("--agent-name", type=str, default=None, help="Label to track which GPU/agent produced a run")
    return parser.parse_args()


def main_entry():
    args = parse_args()
    sweep_id = args.sweep_id
    sweep_config = build_sweep_config()

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project)
        print(f"[sweep] created sweep: {sweep_id}")
    else:
        print(f"[sweep] reusing existing sweep: {sweep_id}")

    if args.create_only:
        print(sweep_id)
        return

    wandb.agent(
        sweep_id,
        function=lambda: _run_inpainting(args),
        project=args.project,
        count=args.count,
    )


if __name__ == "__main__":
    main_entry()
