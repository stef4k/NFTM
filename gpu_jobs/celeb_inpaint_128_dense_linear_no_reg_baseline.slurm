#!/bin/bash
#SBATCH --job-name=celeb128_dense_base
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# ----------------- Environment setup -----------------
source ~/.bashrc
: "${WORKDIR:=$HOME}"
cd "$WORKDIR"
conda activate nftm
cd NFTM

mkdir -p logs runs "$WORKDIR/.cache/torch/hub/checkpoints"

echo "=== Running CelebA-HQ dense controller baseline (128x128) ==="
echo "Job ID: $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# ----------------- Run the experiment -----------------
python image_inpainting.py \
  --train_dataset celebahq \
  --benchmark celebahq \
  --img_size 128 \
  --controller dense \
  --width 48 \
  --epochs 20 \
  --batch_size 64 \
  --K_train 10 \
  --K_eval 10 \
  --pmin 0.25 \
  --pmax 0.5 \
  --block_prob 0.5 \
  --corr_clip 0 \
  --noise_std 0.3 \
  --tv_weight 0 \
  --contract_w 0 \
  --weight_decay 0 \
  --no_curriculum \
  --fixed_K \
  --save_dir runs/celeb_inpaint_128_dense_linear_no_reg_baseline

echo "=== DONE ==="
