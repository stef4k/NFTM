#!/bin/bash
#SBATCH --job-name=unet_parallel
#SBATCH --output=logs/unet_parallel_%A_%a.out
#SBATCH --error=logs/unet_parallel_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --array=0-7          # 4 parallel agents (requires 4 GPUs)

# ---- Environment setup ----
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

SWEEP_FILE="unet_parallel_sweep_${SLURM_ARRAY_JOB_ID}.txt"

# ---- Create sweep once, only task 0 ----
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    echo "=== Creating UNET sweep (parallel version) ==="
    SWEEP_ID=$(python inpainting_grid_search_sweep_a100.py \
                    --create-only \
                    --controller unet \
                    --project unet_parallel_sweep | tail -n 1)
    echo "Sweep ID: $SWEEP_ID"
    echo $SWEEP_ID > $SWEEP_FILE
else
    while [[ ! -s $SWEEP_FILE ]]; do
        echo "[agent $SLURM_ARRAY_TASK_ID] Waiting for sweep file..."
        sleep 3
    done
    SWEEP_ID=$(cat $SWEEP_FILE)
fi

echo "[agent $SLURM_ARRAY_TASK_ID] Using sweep ${SWEEP_ID}"

# ---- Run 4 parallel GPU agents ----
python inpainting_grid_search_sweep_a100.py \
        --sweep-id "${SWEEP_ID}" \
        --controller unet \
        --count 5 \
        --agent-name "agent_${SLURM_ARRAY_TASK_ID}" \
        --run-group "unet_parallel_${SLURM_ARRAY_JOB_ID}" \
        --name-prefix "unet_parallel" \
        --project unet_parallel_sweep