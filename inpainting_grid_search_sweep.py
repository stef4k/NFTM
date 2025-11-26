import os, sys, itertools, wandb
from image_inpainting import main

# Use shared Torch cache
torch_cache = os.path.join(os.environ.get("WORKDIR", os.getcwd()), ".cache", "torch")
os.environ["TORCH_HOME"] = torch_cache
os.makedirs(os.path.join(torch_cache, "hub", "checkpoints"), exist_ok=True)
print(f"[setup] Using cached Torch models from: {os.environ['TORCH_HOME']}")

fixed_config = {
    "controller": "dense",
    "img_size": 32,
    "benchmark": "celebahq",
    "train_dataset": "celebahq"
}

param_grid = {
    "epochs": [10, 15, 20, 25, 30, 35],
    "K_train": [1, 2, 5, 10, 15, 20, 25],
    "K_eval": [5, 10, 20, 25, 30, 35, 40, 45],
}

for epochs, k_train, k_eval in itertools.product(
    param_grid["epochs"], param_grid["K_train"], param_grid["K_eval"]
):
    config = {**fixed_config, "epochs": epochs, "K_train": k_train, "K_eval": k_eval}
    run_name = (
        f"_ep{epochs}_ktr{str(k_train)}_kev{str(k_eval)}"
        f"{config['controller']}_img{config['img_size']}"
        
    )
    save_dir = os.path.join("out", run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=== Starting run: {run_name} ===\n")

    run = wandb.init(
        project="image_inpainting_sweep",
        config=config,
        reinit=True,
        name=run_name,
    )

    sys.argv = [
        "image_inpainting.py",
        "--img_size", str(config["img_size"]),
        "--train_dataset", str(config["train_dataset"]),
        "--benchmark", str(config["benchmark"]),
        "--epochs", str(epochs),
        "--K_train", str(k_train),
        "--K_eval", str(k_eval),
        "--controller", config["controller"],
        "--save_metrics",
        "--use_wandb",
        "--save_dir", save_dir,
    ]


    main()
    run.finish()
    print(f"\n=== Finished run: {run_name} ===\n")
