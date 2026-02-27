#!/bin/bash
#SBATCH --job-name=sidd_test13
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
echo "Config: overlap tiles + guard+gate+linear + mild beta anneal"

cd /gpfs/workdir/balazsk/NFTM
mkdir -p logs out

source ~/.bashrc
conda activate nftm

SAVE_DIR="out/sidd_denoise_test13_overlap_guard"
mkdir -p "$SAVE_DIR"
cat > "${SAVE_DIR}/command.txt" <<'EOF'
python sidd_denoising.py \
  --train_dataset sidd --benchmark sidd \
  --sidd_root /gpfs/workdir/balazsk/NFTM/benchmarks/SIDD/SIDD_Medium_Srgb \
  --patch_size 64 --patch_stride 32 \
  --epochs 12 \
  --batch_size 48 \
  --K_train 10 --K_eval 3 \
  --beta_start 0.18 --beta_max 0.60 --beta_anneal 0.03 --beta_eval_bonus 0.05 \
  --corr_clip 0.09 \
  --contract_w 0.0008 \
  --tv_weight 0.00002 \
  --guard_in_train \
  --gate \
  --step_loss linear \
  --eval_every 1 --eval_max_batches 120 \
  --save_epoch_progress \
  --sidd_viz_samples 100 --sidd_viz_pngs 10 \
  --sidd_viz_max_steps 8 \
  --sidd_limit_tiles 0 \
  --sidd_epoch_tiles 8000 \
  --save_metrics \
  --save_dir out/sidd_denoise_test13_overlap_guard \
  --sidd_save_full_images \
  --sidd_full_max_images 30 \
  --sidd_full_batch 24 \
  --sidd_full_tile_pad 16 \
  --seed 13
EOF
echo "[run] command saved to ${SAVE_DIR}/command.txt"

run_start_epoch="$(date +%s)"
SECONDS=0

python sidd_denoising.py \
  --train_dataset sidd --benchmark sidd \
  --sidd_root /gpfs/workdir/balazsk/NFTM/benchmarks/SIDD/SIDD_Medium_Srgb \
  --patch_size 64 --patch_stride 32 \
  --epochs 12 \
  --batch_size 48 \
  --K_train 10 --K_eval 3 \
  --beta_start 0.18 --beta_max 0.60 --beta_anneal 0.03 --beta_eval_bonus 0.05 \
  --corr_clip 0.09 \
  --contract_w 0.0008 \
  --tv_weight 0.00002 \
  --guard_in_train \
  --gate \
  --step_loss linear \
  --eval_every 1 --eval_max_batches 120 \
  --save_epoch_progress \
  --sidd_viz_samples 100 --sidd_viz_pngs 10 \
  --sidd_viz_max_steps 8 \
  --sidd_limit_tiles 0 \
  --sidd_epoch_tiles 8000 \
  --save_metrics \
  --save_dir "$SAVE_DIR" \
  --sidd_save_full_images \
  --sidd_full_max_images 30 \
  --sidd_full_batch 24 \
  --sidd_full_tile_pad 16 \
  --seed 13

run_end_epoch="$(date +%s)"
elapsed_s="$((run_end_epoch - run_start_epoch))"
echo "Run end   (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
echo "Total wall-clock time: ${elapsed_s}s (${SECONDS}s by bash SECONDS)"
printf "Total wall-clock (hh:mm:ss): %02d:%02d:%02d\n" "$((elapsed_s/3600))" "$(((elapsed_s%3600)/60))" "$((elapsed_s%60))"
