#!/bin/bash
#SBATCH --job-name=inpainting_sweep
#SBATCH --output=logs/inpainting_bayes_%A_%a.out
#SBATCH --error=logs/inpainting_bayes_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1               # one GPU per agent
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --array=0-3                # === RUN 4 PARALLEL AGENTS ===

source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints

export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

# Only the FIRST array task creates the sweep
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    echo "=== Creating sweep ==="
    SWEEP_ID=$(python inpainting_grid_search_sweep.py \
                  --create-only \
                  --project image_inpainting_sweep | tail -n 1)
    echo "Sweep ID: $SWEEP_ID"
    echo $SWEEP_ID > sweep_id.txt
else
    while [ ! -f sweep_id.txt ]; do
        echo "Waiting for sweep_id.txt..."
        sleep 5
    done
    SWEEP_ID=$(cat sweep_id.txt)
fi

echo "[agent $SLURM_ARRAY_TASK_ID] Using sweep ${SWEEP_ID}"

python inpainting_grid_search_sweep.py \
        --sweep-id "${SWEEP_ID}" \
        --count 4 \
        --run-group "bayes_${SLURM_ARRAY_JOB_ID}" \
        --agent-name "agent_${SLURM_ARRAY_TASK_ID}" \
        --project image_inpainting_sweep
