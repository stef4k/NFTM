#!/bin/bash
#SBATCH --job-name=sidd_test9
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
echo "Run start (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
echo "Config: conservative beta+clip, guard in train, linear step loss"

cd /gpfs/workdir/balazsk/NFTM
mkdir -p logs out

source ~/.bashrc
conda activate nftm

SAVE_DIR="out/sidd_denoise_check9"
mkdir -p "$SAVE_DIR"
cat > "${SAVE_DIR}/command.txt" <<'EOF'
python image_inpainting.py \
  --train_dataset sidd --benchmark sidd \
  --sidd_root /gpfs/workdir/balazsk/NFTM/benchmarks/SIDD/SIDD_Medium_Srgb \
  --patch_size 64 --patch_stride 64 \
  --controller unet \
  --epochs 10 \
  --batch_size 64 \
  --K_train 6 --K_eval 12 \
  --beta_start 0.20 --beta_max 0.50 --beta_eval_bonus 0.00 \
  --corr_clip 0.08 \
  --contract_w 0.0005 \
  --tv_weight 0.0 \
  --guard_in_train \
  --step_loss linear \
  --eval_every 1 --eval_max_batches 50 \
  --save_epoch_progress \
  --sidd_viz_samples 50 --sidd_viz_pngs 10 \
  --sidd_viz_max_steps 8 \
  --sidd_limit_tiles 5000 \
  --sidd_epoch_tiles 2048 \
  --save_dir out/sidd_denoise_check9 \
  --sidd_save_full_images \
  --sidd_full_max_images 30 \
  --sidd_full_batch 32 \
  --sidd_full_tile_pad 16
EOF
echo "[run] command saved to ${SAVE_DIR}/command.txt"

run_start_epoch="$(date +%s)"
SECONDS=0

python image_inpainting.py \
  --train_dataset sidd --benchmark sidd \
  --sidd_root /gpfs/workdir/balazsk/NFTM/benchmarks/SIDD/SIDD_Medium_Srgb \
  --patch_size 64 --patch_stride 64 \
  --controller unet \
  --epochs 10 \
  --batch_size 64 \
  --K_train 6 --K_eval 12 \
  --beta_start 0.20 --beta_max 0.50 --beta_eval_bonus 0.00 \
  --corr_clip 0.08 \
  --contract_w 0.0005 \
  --tv_weight 0.0 \
  --guard_in_train \
  --step_loss linear \
  --eval_every 1 --eval_max_batches 50 \
  --save_epoch_progress \
  --sidd_viz_samples 50 --sidd_viz_pngs 10 \
  --sidd_viz_max_steps 8 \
  --sidd_limit_tiles 5000 \
  --sidd_epoch_tiles 2048 \
  --save_dir "$SAVE_DIR" \
  --sidd_save_full_images \
  --sidd_full_max_images 30 \
  --sidd_full_batch 32 \
  --sidd_full_tile_pad 16

run_end_epoch="$(date +%s)"
elapsed_s="$((run_end_epoch - run_start_epoch))"
echo "Run end   (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
echo "Total wall-clock time: ${elapsed_s}s (${SECONDS}s by bash SECONDS)"
printf "Total wall-clock (hh:mm:ss): %02d:%02d:%02d\n" "$((elapsed_s/3600))" "$(((elapsed_s%3600)/60))" "$((elapsed_s%60))"
