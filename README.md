# NFTM

Neural Field Turing Machine experiments implementing differentiable cellular automata and diffusion processes.

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<user>/NFTM.git
   cd NFTM
   ```
2. **Install dependencies** (Python 3.8+)
   ```bash
   pip install torch torchvision numpy matplotlib pillow imageio
   ```

## Rule 110
Train a stateless controller to reproduce Rule 110 and visualize the rollout.

```bash
python3 rule110_nftm.py --train_mode truth_table
```

The script writes PNG and GIF visualizations to `results_rule110/`. Use `--train_mode gt_rollout` to learn from ground‑truth sequences instead.

## Game of Life
Simulate Conway's Game of Life using a pretrained NFTM controller.

```bash
python3 game_of_life.py
```

GIFs for glider, blinker and random patterns are saved as `life_<mode>.gif` in the working directory.

## Heat Equation
Differentiable solver for the 2‑D heat equation. Only the following modes are documented here:

### Test α mode
Checks whether the model can recover known diffusion coefficients with a small hyperparameter search.
```bash
python3 heat_eq.py --mode test_alpha
```
### Variable α mode
Learns a spatially varying diffusion field.
```bash
python3 heat_eq.py --mode variable
```
Outputs for each mode are stored in `results_<mode>_<size>/`.

### Generalization Beyond Training Horizon (Variable α Mode)
You can test if the model generalizes to longer rollouts than it was trained on by using the `--eval_timesteps` flag:

```bash
python3 heat_eq.py --mode variable --timesteps 10 --eval_timesteps 20 --epochs_a 50 --epochs_b 50 --size 32 --batch_size 4
```

- `--timesteps`: number of steps used during training
- `--eval_timesteps`: number of steps for evaluation/rollout (can be longer than training)
- The output PSNR curve and visualizations will show how well the model extrapolates to longer horizons.

## Image Inpainting
Iteratively inpaint masked CIFAR‑10 images. Recommended command:
```bash
python3 image_inpainting.py --epochs 10 --batch_size 128 --K_train 10 --K_eval 13 --beta_start 0.28 --beta_max 0.6 --beta_anneal 0.03 --beta_eval_bonus 0.05 --corr_clip 0.1 --tv_weight 0.01 --pmin 0.25 --pmax 0.5 --block_prob 0.5 --noise_std 0.3 --width 48 --save_metrics --loss homo --homo_sigma 0.1 --guard_in_train --save_dir runs/train_guarded
```
The CIFAR‑10 dataset is downloaded automatically and results are saved under `runs/train_guarded/`.

## Ablation Experiments
Run a suite of image inpainting ablations comparing loss modes and hyperparameters.
```bash
python3 run_experiments.py --runs_dir runs/ablations
```
Each experiment's metrics are saved under `runs/ablations/<experiment_name>/` along with a summary plot and CSV in `runs/ablations/`.

