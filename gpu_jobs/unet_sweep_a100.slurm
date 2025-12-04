#!/bin/bash
#SBATCH --job-name=unet_sweep_a100
#SBATCH --output=logs/unet_sweep_%A.out
#SBATCH --error=logs/unet_sweep_%A.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# ---- Environment ----
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

SWEEP_FILE="unet_sweep_${SLURM_JOB_ID}.txt"

# ---- Create sweep once ----
if [[ ! -f $SWEEP_FILE ]]; then
    echo "=== Creating UNET sweep ==="
    SWEEP_ID=$(python inpainting_grid_search_sweep_a100.py \
                    --create-only \
                    --project unet_inpainting_sweep \
                    --controller unet \
                    | tail -n 1)
    echo $SWEEP_ID > $SWEEP_FILE
else
    SWEEP_ID=$(cat $SWEEP_FILE)
fi

echo "[UNET A100] Sweep: ${SWEEP_ID}"

# ---- Run sweep agent (sequential configs) ----
python inpainting_grid_search_sweep_a100.py \
        --sweep-id "${SWEEP_ID}" \
        --controller unet \
        --count 1000 \
        --run-group "unet_${SLURM_JOB_ID}" \
        --name-prefix "unet" \
        --project unet_inpainting_sweep
