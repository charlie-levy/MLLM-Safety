#!/bin/bash
#SBATCH --job-name=preview_figstep_llava_cot_blur5
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.log

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ch169788/llava_cot_eval

python code/run_eval.py \
    --model       llava_cot \
    --dataset     figstep \
    --corrupt     gaussian_blur \
    --pct         5 \
    --n           10 \
    --out         results/preview/figstep_llava_cot_blur5.json \
    --save_images results/preview/img_figstep_blur5