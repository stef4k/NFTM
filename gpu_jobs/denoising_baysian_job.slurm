#!/bin/bash
#SBATCH --job-name=cifar_denoising_bayes
#SBATCH --output=logs/cifar_denoising_bayes_%A_%a.out
#SBATCH --error=logs/cifar_denoising_bayes_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --array=0-7

# ---- Environment setup ----
source ~/.bashrc
cd "$WORKDIR"
conda activate nftm
cd NFTM

mkdir -p logs out "$WORKDIR/.cache/torch/hub/checkpoints"

export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

SWEEP_FILE="cifar_denoising_bayes_${SLURM_ARRAY_JOB_ID}.txt"

# Keep naming consistent
PROJECT="cifar_denoising_bayes_fixed_seed"
RUN_GROUP="dense_parallel_${SLURM_ARRAY_JOB_ID}"
NAME_PREFIX="dense_parallel"
COUNT_PER_AGENT=50

# ---- Create sweep once, only task 0 ----
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    echo "=== Creating sweep ==="
    python denoising_baysian_sweep.py --create-only --project "${PROJECT}" | tee logs/sweep_create_${SLURM_ARRAY_JOB_ID}.log

    # Extract sweep id robustly: assume sweep id is the last non-empty line printed
    SWEEP_ID="$(grep -Eo '[a-zA-Z0-9]{8,}' logs/sweep_create_${SLURM_ARRAY_JOB_ID}.log | tail -n 1)"

    if [[ -z "${SWEEP_ID}" ]]; then
        echo "ERROR: Failed to extract SWEEP_ID from sweep creation log."
        exit 1
    fi

    echo "Sweep ID: ${SWEEP_ID}"
    echo "${SWEEP_ID}" > "${SWEEP_FILE}"
else
    while [[ ! -s "${SWEEP_FILE}" ]]; do
        echo "[agent ${SLURM_ARRAY_TASK_ID}] Waiting for sweep file..."
        sleep 3
    done
    SWEEP_ID="$(cat "${SWEEP_FILE}")"
fi

echo "[agent ${SLURM_ARRAY_TASK_ID}] Using sweep ${SWEEP_ID}"

# ---- Run one GPU agent per array task ----
python denoising_baysian_sweep.py \
    --sweep-id "${SWEEP_ID}" \
    --project "${PROJECT}" \
    --count "${COUNT_PER_AGENT}" \
    --agent-name "agent_${SLURM_ARRAY_TASK_ID}" \
    --run-group "${RUN_GROUP}" \
    --name-prefix "${NAME_PREFIX}" \
    --save-root "out"
