#!/bin/bash
#SBATCH -J nftm_inpaint
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1      # <-- THIS was missing
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/inpaint.%j.out
#SBATCH -e logs/inpaint.%j.err
#SBATCH --comment="FiftyOne inpaint viz"

set -euo pipefail

# --- user-configurable bits ---
VENV_DIR="${HOME}/venvs/nftm"
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
SAVE_DIR="out/cruche_job_${SLURM_JOB_ID:-manual}"
FO_PORT=5151
FO_ADDR="0.0.0.0"   # keep 0.0.0.0 for SSH port-forward
FO_MAX=6

module purge
# module load python/3.x  # uncomment/adjust to your site

source "${VENV_DIR}/bin/activate"

export FIFTYONE_DATABASE_DIR="${SCRATCH:-$HOME}/fiftyone_db"
export OMP_NUM_THREADS=8

cd "${WORKDIR}"
mkdir -p logs "${SAVE_DIR}"

python -u image_inpainting.py \
  --device gpu \
  --epochs 1 \
  --batch_size 16 \
  --save_epoch_progress \
  --save_dir "${SAVE_DIR}" \
  --fo_log \
  --fo_max_samples "${FO_MAX}" \
  --fo_address "${FO_ADDR}" \
  --fo_port "${FO_PORT}"

echo "[done] outputs in ${SAVE_DIR}"
