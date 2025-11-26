#!/bin/bash
#SBATCH --job-name=inpainting_sweep
#SBATCH --output=logs/inpainting_bayes_%j.out
#SBATCH --error=logs/inpainting_bayes_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4          # request 8 GPUs
#SBATCH --ntasks=4                # one task per GPU
#SBATCH --gpus-per-task=1         # exactly 1 GPU per task
#SBATCH --mem=64G
#SBATCH --partition=gpu

# ---- Environment setup ----
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

# Make sure logs and cache folders exist
mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints

# Set safer defaults for multi-GPU sweeps
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap
export NCCL_P2P_DISABLE=1


echo "=== Creating sweep ==="
SWEEP_ID=$(python inpainting_grid_search_sweep.py --create-only --project image_inpainting_sweep | tail -n 1)

echo "Sweep ID inside SLURM: '${SWEEP_ID}'"

RUNS_PER_AGENT=${RUNS_PER_AGENT:-4}
GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[agents] starting ${GPUS} agents | runs/agent=${RUNS_PER_AGENT}"

for ((i=0; i<GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i srun --exclusive --gres=gpu:1 --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python inpainting_grid_search_sweep.py \
      --sweep-id "${SWEEP_ID}" \
      --count "${RUNS_PER_AGENT}" \
      --run-group "bayes_${SLURM_JOB_ID}" \
      --agent-name "gpu${i}" \
      --project image_inpainting_sweep &
done

wait
