#!/bin/bash
#SBATCH --job-name=inpainting_sweep
#SBATCH --output=logs/inpainting_grid_sweep_%j.out
#SBATCH --error=logs/inpainting_grid_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --qos=normal

# ---- Environment setup ----
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

# Make sure logs and cache folders exist
mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints

# Run the Python sweep script
python inpainting_grid_search_sweep.py