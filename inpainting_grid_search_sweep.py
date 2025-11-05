import os, sys, itertools, wandb
from image_inpainting import main

# Use shared Torch cache
torch_cache = os.path.join(os.environ.get("WORKDIR", os.getcwd()), ".cache", "torch")
os.environ["TORCH_HOME"] = torch_cache
os.makedirs(os.path.join(torch_cache, "hub", "checkpoints"), exist_ok=True)
print(f"[setup] Using cached Torch models from: {os.environ['TORCH_HOME']}")

param_grid = {
    "controller": ["dense", "unet"],
    "img_size": [32],
    "epochs": [10,15,20,25,30],
    "K_train": [1,2,5,7,10],
    "K_eval": [10,20,25,30,35,40],
    "beta_start": [0.28, 0.25, 0.3],
    "beta_max": [0.6],
    "beta_anneal": [0.03],
    "beta_eval_bonus": [0.05],
    "corr_clip": [0.05, 0.1, 0.2],
    "tv_weight": [0.01],
    "contract_w": [1e-3, 5e-4],
    "lr": [2e-3, 1e-3],
    "batch_size": [64, 128],
    "pmin": [0.25],
    "pmax": [0.5],
    "block_prob": [0.5],
    "noise_std": [0.3],
    "guard_in_train": [False, True],
}

for (
    controller, img_size, epochs, k_train, k_eval, beta_start, beta_max,
    beta_anneal, beta_eval_bonus, corr_clip, tv_weight, contract_w,
    lr, batch_size, pmin, pmax, block_prob, noise_std, guard_in_train,
) in itertools.product(
    param_grid["controller"], param_grid["img_size"], param_grid["epochs"], param_grid["K_train"],
    param_grid["K_eval"], param_grid["beta_start"],param_grid["beta_max"], param_grid["beta_anneal"],
    param_grid["beta_eval_bonus"], param_grid["corr_clip"], param_grid["tv_weight"], param_grid["contract_w"],
    param_grid["lr"], param_grid["batch_size"], param_grid["pmin"], param_grid["pmax"],
    param_grid["block_prob"], param_grid["noise_std"], param_grid["guard_in_train"],
):
    run_name = (
            f"{controller}_img{img_size}_ktr{str(k_train)}_kev{str(k_eval)}"
            f"_beta{str(beta_start)}_clip{str(corr_clip)}"
            f"_guard{int(guard_in_train)}"
        )   
    save_dir = os.path.join("out", run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=== Starting run: {run_name} ===\n")

    run = wandb.init(
        project="image_inpainting_sweep",
        config={
            "controller": controller,
            "img_size": img_size,
            "epochs": epochs,
            "K_train": k_train,
            "K_eval": k_eval,
            "beta_start": beta_start,
            "beta_max": beta_max,
            "beta_anneal": beta_anneal,
            "beta_eval_bonus": beta_eval_bonus,
            "corr_clip": corr_clip,
            "tv_weight": tv_weight,
            "contract_w": contract_w,
            "lr": lr,
            "batch_size": batch_size,
            "pmin": pmin,
            "pmax": pmax,
            "block_prob": block_prob,
            "noise_std": noise_std,
            "guard_in_train": guard_in_train,
        },
        reinit=True,
        name=run_name,
    )

    sys.argv = [
        "image_inpainting.py",
        "--img_size", str(img_size),
        "--train_dataset", "cifar",
        "--benchmark", "cifar",
        "--epochs", str(epochs),
        "--K_train", str(k_train),
        "--K_eval", str(k_eval),
        "--beta_start", str(beta_start),
        "--beta_max", str(beta_max),
        "--beta_anneal", str(beta_anneal),
        "--beta_eval_bonus", str(beta_eval_bonus),
        "--corr_clip", str(corr_clip),
        "--tv_weight", str(tv_weight),
        "--contract_w", str(contract_w),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--pmin", str(pmin),
        "--pmax", str(pmax),
        "--block_prob", str(block_prob),
        "--noise_std", str(noise_std),
        "--controller", controller,
        "--save_metrics",
        "--use_wandb",
        "--save_dir", save_dir,
    ]

    if guard_in_train:
        sys.argv.append("--guard_in_train")

    main()
    run.finish()
    print(f"\n=== Finished run: {run_name} ===\n")