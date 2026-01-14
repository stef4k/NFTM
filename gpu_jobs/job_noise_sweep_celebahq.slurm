#!/bin/bash
#SBATCH --job-name=nftm_unet_celebahq_noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=finished_jobs/out_nftm_unet_celebahq_%j.log
#SBATCH --error=finished_jobs/err_nftm_unet_celebahq_%j.log

# ----------------------------
# Environment
# ----------------------------

source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM


# ----------------------------
# Reproducibility
# ----------------------------
export PYTHONHASHSEED=0

# ----------------------------
# Run
# ----------------------------
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
  --eval_noise_sweep
