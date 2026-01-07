#!/bin/bash
#SBATCH --job-name=dense_single_64_fixed_seed
#SBATCH --output=logs/dense_single_img64_%A.out
#SBATCH --error=logs/dense_single_img64_%A.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# ----------------- Environment setup -----------------
source ~/.bashrc
cd $WORKDIR
conda activate nftm
cd NFTM

mkdir -p logs out $WORKDIR/.cache/torch/hub/checkpoints

echo "=== Running single dense inpainting experiment ==="
echo "Job ID: $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# ----------------- Run the experiment -----------------
python image_inpainting.py \
    --img_size 64 \
    --train_dataset celebahq \
    --benchmark celebahq \
    --controller dense \
    --guard_in_train \
    --save_metrics \
    --batch_size 128 \
    --save_dir out/dense_single_ep50_ktr23_kev24_celebahq_img64 \
    --controller dense \
    --seed 0 --device cuda --save_metrics \
    --beta_eval_bonus 0.11014919643164978 --beta_max 0.8964916582414793 \
    --beta_start 0.4427944180580322 \
    --contract_w 0.0004217134973337905 \
    --corr_clip 0.23531494426062569 --epochs 30 --K_eval 20 --K_train 20 \
    --tv_weight 0.004093171742958494


echo "=== DONE ==="