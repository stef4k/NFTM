# NFTM

**NFTM for Computer Vision: Adaptive Inpainting and Denoising**
Research project carried out in the Big Data Research Project (BDRP) course of the BDMA master by Kristóf Balázs, Stefanos Kypritidis and Nicola Mambelli, in collaboration with PhD student Akash Malhotra (Amadeus) and advised by Nacera Seghouani (CentraleSupélec).
 
This project builds on the Neural Field Turing Machine (NFTM), introduced by Malhotra and Seghouani in [*Neural Field Turing Machine: A Differentiable Spatial Computer*](https://arxiv.org/abs/2509.03370) (2025). The paper presents NFTM as a general framework where a neural controller repeatedly reads and updates a spatial field through local operations, and shows CIFAR-10 image inpainting as one of its applications. We started from that inpainting pipeline and its codebase, and extended it to higher resolutions, new datasets and a second task, denoising. Most restoration models predict the clean image in a single forward pass. In NFTM, both tasks are framed as iterative reconstruction: starting from the corrupted image, a small learned controller looks at the current estimate and predicts a correction, and the image is refined over K steps. For inpainting, the known pixels are clamped to their observed values so only the missing region is updated; for denoising, no pixel is treated as known and the update applies to the whole image. Both tasks share the same code path and are selected through the masking flags.
 
The controller is either a small 3-layer convolutional network (`dense`) or a compact U-Net (`unet`). The pipeline from the original paper surrounds it with several stabilisation mechanisms: a per-pixel gate, a damping factor β with annealing, per-step clipping, an energy-based guard with backtracking, a contractive penalty and a rollout curriculum. We studied each of these components through ablations on CIFAR-10 and CelebA-HQ, and compared NFTM against a one-shot U-Net with a similar number of parameters, a non-learned TV-L1 method and two small DDPM diffusion models. Set12 and CBSD68 are used as evaluation-only benchmarks, and SIDD is used to test denoising on real smartphone camera noise. Results are reported with PSNR, SSIM, LPIPS, FID and KID, both at the final step and after every refinement step. All heavy training and evaluation runs were done on V100 and A100 GPUs of the Ruche HPC cluster (Mésocentre Paris-Saclay), with jobs submitted through SLURM (see `gpu_jobs/`), while Google Colab was used for quick tests and debugging.
 
Our main finding is that performance comes from the iterative learned correction itself, not from the stabilisation mechanisms of the original pipeline. The best configurations drop the gate, β damping, clipping, the guard and backtracking, and use a fixed rollout with the same number of steps in training and evaluation. With this simpler setup, NFTM outperforms the one-shot U-Net, TV-L1 and diffusion baselines on all five metrics for inpainting, reaching 28.79 dB PSNR on CIFAR-10 and 30.63 dB PSNR (SSIM 0.951) on CelebA-HQ at 128×128. For denoising with additive Gaussian noise, the U-Net controller reaches 30.84 dB on CelebA-HQ, and a tile-based version trained on SIDD reaches 40.86 dB, which shows that the approach carries over to real sensor noise. The main limitations are sensitivity to hyperparameters and slower inference than single-pass models, since each image needs several controller evaluations.
 
This repository contains the NFTM implementation, the baselines and the scripts we used for training, benchmarking and ablations. Find:
- [The final report](https://github.com/stef4k/NFTM/blob/main/BDRP_report.pdf)
- [The final presentation](https://github.com/stef4k/NFTM/blob/main/BDRP_presentation.pdf)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/akashjorss/NFTM.git
   cd NFTM
   ```
2. **Install dependencies** (Python 3.8+)
   ```bash
   pip install torch torchvision numpy matplotlib pillow imageio
   ```

## Image Inpainting

Repository structure:
```
NFTM/
├─ image_inpainting.py                  # Main CLI: train/eval one run (pyramid or single-scale)
├─ run_experiments.py                   # Batch launcher + result aggregator for multiple runs
├─ run-experiments-sweep.py             # Hyperparameter sweep helper (local)
├─ inpainting_grid_search_sweep.py      # Exhaustive hyperparameter sweep visualized in weights and biases
│
├─ nftm_inpaint/                        # Inpainting related modules
│  ├─ __init__.py
│  ├─ controller.py                     # TinyController & UNetController wrappers
│  ├─ engine.py                         # Train/eval engines, descent guard, energy, metrics eval
│  ├─ rollout.py                        # NFTM step, multi-scale pyramid rollout helpers
│  ├─ data_and_viz.py                   # Datasets, transforms, masking, plotting/PNG/GIF
│  ├─ metrics.py                        # PSNR/SSIM/LPIPS + FID/KID wrappers
│  ├─ unet_model.py                     # TinyUNet backbone used by UNetController
│  ├─ train_unet.py                     # Baseline UNet trainer (supervised)
│  └─ eval_unet.py                      # Baseline UNet evaluator/metrics
│
├─ drivers/
│  └─ run_all_inpainting.py
│
├─ baselines/
│  └─ inpainting/
│     └─ tvl1_baseline.py              # TV-L1 primal-dual inpainting baseline
├─ docs/
│  └─ pyramid_trick.md                 # Notes on the multi-scale (pyramid) rollout
gpu_jobs/
|  ├─ gpu_job.slurm                       # Single-run job: train/eval image_inpainting.py on GPU
|  └─ inpainting_grid_sweep.slurm         # Large grid-search sweep using inpainting_grid_search_sweep.py
```

Quick start (single-scale, CIFAR-10 @ 32×32):
```bash
python3 image_inpainting.py \
  --epochs 10 \
  --batch_size 128 \
  --K_train 10 \
  --K_eval 13 \
  --beta_start 0.28 --beta_max 0.6 --beta_anneal 0.03 --beta_eval_bonus 0.05 \
  --corr_clip 0.1 --tv_weight 0.01 \
  --pmin 0.25 --pmax 0.5 --block_prob 0.5 --noise_std 0.3 \
  --width 48 \
  --save_metrics \
  --save_dir runs/baseline
```
The CIFAR‑10 dataset is downloaded automatically and results are saved under `runs/baseline/`.

Artifacts:
- `runs/.../steps/progress_epoch_<E>.png`: step-by-step grid for the first test batch each epoch (if --save_epoch_progress).
- `runs/.../final/progress_final.png`: final step grid after training.
- `runs/.../psnr_curve.png`, `ssim_curve.png`, `lpips_curve.png`.
- `runs/.../metrics.json` (if `--save_metrics`), including PSNR/SSIM/LPIPS and FID/KID.
- Add `--use_wandb` to stream metrics to Weights & Biases.

## Inpainting: Flags reference

### Data & I/O

- `--train_dataset` {cifar,celebahq} (default: cifar):  dataset used for training.
- `--benchmark` {cifar,set12,cbsd68,celebahq} (default: cifar):  dataset used for evaluation. See the [Benchmarks](#benchmarks) section.

- `--img_size` {32,64} (default: 32):  working resolution. If you use a pyramid, the largest size must equal this.

- `--save_dir` *PATH* (default: out):  where to write plots and artifacts.

- `--save_metrics`: also saves metrics.json (+ *_curve.npy) and computes FID/KID.

- `--save_epoch_progress`: saves per-epoch step grids under save_dir/steps/.

- `--viz_scale` *FLOAT* (default: 1.0):  upsample factor for the saved PNG/GIF (visualization only).

- `--use_wandb`: log to Weights & Biases.

- `--seed` *INT* (default: 0):  RNG seed.

- `--device` *STR* (default: auto cuda/cpu): force device.

### Model / Controller

- `--controller` {dense,unet} (default: dense): controller architecture.

- `--width` *INT* (default: 48): channel width for the dense controller.

- `--unet_base` *INT* (default: 10): base channels for the UNet controller (auto-tuned to match param count when possible).

### Optimization

- `--epochs` *INT* (default: 10)

- `--batch_size` *INT* (default: 128)

- `--lr` *FLOAT* (default: 2e-3): learning rate.

- `--weight_decay` *FLOAT* (default: 1e-4) 

- `--tv_weight` *FLOAT* (default: 0.01): total variation regularizer weight.

- `--contract_w` *FLOAT* (default: 1e-3): contractive penalty weight.
### Rollout & step sizes

- `--K_train` *INT* (default: 8): maximum rollout depth during training; actual steps are sampled per batch by a curriculum (shorter early, longer later).

- `--K_eval` *INT* (default: 12): rollout depth during evaluation/visualization.

- `--beta_start` *FLOAT* (default: 0.28): initial step size $\mathcal{\beta}$.

- `--beta_max` *FLOAT* (default: 0.6): cap on $\mathcal{\beta}$ during training.

- `--beta_anneal` *FLOAT* (default: 0.03): per-epoch $\mathcal{\beta}$ increment.
- `--beta_eval_bonus` *FLOAT* (default: 0.05): extra $\mathcal{\beta}$ added at eval time.
- `--corr_clip` *FLOAT* (default: 0.1): max per-step correction magnitude before gating.

Note: With a pyramid, both $\mathcal{\beta}$ and corr_clip are scaled at each level by (final_size / level_size) so coarse levels take bolder steps.
### Masking / corruption

- `--pmin` *FLOAT* (default: 0.25): min missing fraction for random pixel drops.

- `--pmax` *FLOAT* (default: 0.5): max missing fraction.

- `--block_prob` *FLOAT* (default: 0.5): probability of adding random square occlusion blocks.

- `--noise_std` *FLOAT* (default: 0.3): noise std used to corrupt unknown pixels.

### Multi-scale (pyramid) rollout

- `--pyramid` "s1,s2,...,S" (default: empty = single-scale): comma-separated sizes from coarse to fine.
Examples: `--pyramid 16,32,64` (requires `--img_size 64`), or `--pyramid 16,32` for 32×32 runs.

- `--pyr_steps` "t1,t2,...,tL" (optional): integer steps per pyramid level that sum to `K_eval`.
If omitted: each coarse level gets 1 step, and the remainder goes to the finest level.

## Ablation Experiments
Run a suite of image inpainting ablations comparing loss modes and hyperparameters.
```bash
python3 run_experiments.py --runs_dir runs/ablations
```
Each experiment's metrics are saved under `runs/ablations/<experiment_name>/` along with a summary plot and CSV in `runs/ablations/`.

### Using the driver script for bulk experiments

The `run_experiments.py` driver orchestrates multiple calls to `image_inpainting.py` with curated hyperparameters. By default it
launches six experiments covering heteroscedastic/homoscedastic losses, rollout length, clipping, and beta schedules.

1. **Choose an output directory** (default: `runs/`):
   ```bash
   python3 run_experiments.py --runs_dir runs/ablations
   ```
   This creates one subfolder per experiment (e.g. `runs/ablations/hetero_baseline/`). Logs, `metrics.json`, and intermediate
   checkpoints are stored there.
2. **Adjust the training budget** if desired with `--epochs` (default: 10). All runs share the same epoch count:
   ```bash
   python3 run_experiments.py --runs_dir runs/ablations --epochs 12
   ```
3. **Point to a custom script** with `--script` when extending the driver to new variants:
   ```bash
   python3 run_experiments.py --script drivers/run_all_inpainting.py
   ```

When the driver finishes, it aggregates results by saving

- `compare_curves.png`: PSNR curves for each experiment plotted on the same axes.
- `summary.csv`: a table containing final metrics, hyperparameters, and the full PSNR trajectory.
- A textual summary printed to the console showing final PSNR values sorted from best to worst.

Failed runs (missing `metrics.json`) are reported in the console log and excluded from the summary artifacts.


### <a name="benchmarks"></a> Extra Benchmarks for inpainting script

To prepare the datasets for benchmarking, follow the steps below:

1. **Create the Benchmarks Folder**
   ```bash
   mkdir -p benchmarks

Then download the datasets:

- CelebAHQ: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256), extract and place in `benchmarks/CelebAHQ/all_images/`

- Set12: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/leweihua/set12-231008), extract and place in `benchmarks/Set12/all_images/`

- CBSD68: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tarekmebrouk/cbsd68), extract only the `train` folder and place it in `benchmarks/CBSD68/all_images/`

- SIDD: Download the dataset from [github](https://abdokamel.github.io/sidd/), download only the `SIDD-Medium Dataset` > `sRGB images only` > `Mirror 1` or directly from this [link](http://130.63.97.225/share/SIDD_Medium_Srgb.zip) and place the unzipped items in `benchmarks/SIDD/`

2. **After setup, run benchmarks using:**

   ```bash
   python image_inpainting.py --benchmark [benchmark_name]
     ```
Available benchmark options: `cifar, set12, cbsd68, celebahq`
For the SIDD dataset run:
   ```bash
   python sidd_denoising.py
   ```
