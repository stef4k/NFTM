#!/bin/bash
#SBATCH --job-name=inpaint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=finished_jobs/out_inpaint_%j.log

source ~/venv/bin/activate
python image_inpainting.py --epochs 2 --device cuda