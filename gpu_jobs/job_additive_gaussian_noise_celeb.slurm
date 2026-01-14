#!/bin/bash
#SBATCH --job-name=unet_single_64_additive_noise
#SBATCH --output=logs/unet_single_img64_additive_noise_%A.out
#SBATCH --error=logs/unet_single_img64_additive_noise_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# ----------------- Environment setup -----------------
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints

echo "=== Running single UNet inpainting experiment with additive noise==="
echo "Job ID: $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# ----------------- Run the experiment -----------------
python image_inpainting.py \
  --train_dataset celebahq \
  --benchmark celebahq \
  --img_size 64 \
  --controller unet \
  --epochs 50 \
  --K_train 17 \
  --K_eval 23 \
  --beta_start 0.5784 \
  --beta_max 0.8722 \
  --beta_eval_bonus 0.077 \
  --contract_w 0.0001 \
  --corr_clip 0.3257 \
  --tv_weight 0.0020 \
  --seed 0 \
  --device cuda \
  --save_metrics \
  --save_epoch_progress \
  --save_dir out/unet_additive_noise_try_elebahq_img64 \
  --gaussian_additive


echo "=== DONE ==="