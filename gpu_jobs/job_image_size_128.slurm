#!/bin/bash
#SBATCH --job-name=unet_single_128
#SBATCH --output=logs/unet_single_img128_%A.out
#SBATCH --error=logs/unet_single_img128_%A.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# ----------------- Environment setup -----------------
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

echo "=== Running single UNet inpainting experiment ==="
echo "Job ID: $SLURM_JOB_ID"

# ----------------- Run the experiment -----------------
python image_inpainting.py \
    --img_size 128 \
    --train_dataset celebahq \
    --benchmark celebahq \
    --epochs 50 \
    --K_train 23 \
    --K_eval 24 \
    --controller unet \
    --beta_start 0.3721035421485907 \
    --beta_max 0.9120364366644372 \
    --beta_eval_bonus 0.07230116021489082 \
    --corr_clip 0.3692418917956467 \
    --tv_weight 0.003326461690307824 \
    --contract_w 0.0002701598741189249 \
    --guard_in_train \
    --save_metrics \
    --batch_size 64

echo "=== DONE ==="