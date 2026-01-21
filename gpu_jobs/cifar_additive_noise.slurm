#!/bin/bash
#SBATCH --job-name=dense_single_32_additive_noise_cifar_denoising 
#SBATCH --output=logs/dense_single_img32_additive_noise_cifar_denoising_%A.out
#SBATCH --error=logs/dense_single_img32_additive_noise_cifar_denoising_%A.err
#SBATCH --partition=gpua100
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

echo "=== Running single dense inpainting experiment with additive noise on cifar==="
echo "Job ID: $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# ----------------- Run the experiment -----------------
python image_inpainting.py \
  --controller dense \
  --beta_eval_bonus 0.1079 \
  --beta_max 0.8800 \
  --beta_start 0.4230 \
  --contract_w 0.0005 \
  --corr_clip 0.3276 \
  --epochs 30 \
  --img_size 32 \
  --K_eval 25 \
  --K_train 22 \
  --tv_weight 0.0050 \
  --seed 0 \
  --train_dataset cifar \
  --pmin 1 \
  --pmax 1 \
  --block_prob 0 \
  --benchmark cifar \
  --device cuda \
  --save_metrics \
  --save_epoch_progress \
  --save_dir out/dense_additive_noise_try_img32_cifar_denoising \
  --gaussian_additive


echo "=== DONE ==="