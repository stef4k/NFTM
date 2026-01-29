#!/bin/bash
#SBATCH --job-name=sidd_test4
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

echo "Host: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Go to your repo
cd /gpfs/workdir/balazsk/NFTM

# Make sure log dirs exist
mkdir -p logs out

source ~/.bashrc
conda activate nftm 

# If you truly use venv instead, then use an absolute path that exists on nodes:
# source /gpfs/workdir/balazsk/venv/bin/activate



  python image_inpainting.py \
  --train_dataset sidd --benchmark sidd \
  --controller unet \
  --sidd_root /gpfs/workdir/balazsk/NFTM/benchmarks/SIDD/SIDD_Medium_Srgb \
  --patch_size 64 --patch_stride 64 \
  --epochs 20 \
  --batch_size 64 \
  --K_train 11 --K_eval 3 \
  --beta_start 0.42 --beta_max 0.95 --beta_eval_bonus 0.01 \
  --corr_clip 0.25 \
  --contract_w 0.0005 \
  --tv_weight 0.005 \
  --eval_every 1 --eval_max_batches 50 \
  --save_epoch_progress \
  --sidd_viz_samples 100 --sidd_viz_pngs 10 \
  --sidd_viz_max_steps 24 \
  --sidd_limit_tiles 10000 \
  --save_dir out/sidd_denoise_check4 \
  --save_epoch_progress \