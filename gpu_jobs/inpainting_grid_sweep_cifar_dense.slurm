#!/bin/bash
#SBATCH --job-name=dense_parallel
#SBATCH --output=logs/dense_parallel_%A_%a.out
#SBATCH --error=logs/dense_parallel_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --array=0-3

# ---- Environment setup ----
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

SWEEP_FILE="dense_parallel_sweep_fixed_seed_cifar${SLURM_ARRAY_JOB_ID}.txt"

# ---- Create sweep once, only task 0 ----
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    echo "=== Creating Dense sweep (parallel version) ==="
    SWEEP_ID=$(python inpainting_grid_search_sweep_a100.py \
                    --create-only \
                    --controller dense \
                    --project dense_parallel_sweep_fixed_seed_cifar | tail -n 1)
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
        --controller dense \
        --count 10 \
        --agent-name "agent_${SLURM_ARRAY_TASK_ID}" \
        --run-group "dense_parallel_${SLURM_ARRAY_JOB_ID}" \
        --name-prefix "dense_parallel" \
        --project dense_parallel_sweep_fixed_seed_cifar