#!/bin/bash
#SBATCH --job-name=judge_beavertails_blur
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
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval

# LLaMA-Guard ASR on the BeaverTails-V blur preview (clean / blur5 / blur10).
# Loads Guard once from the HF cache (--mem=80G keeps it off the mmap-thrash path).
python code/judge_grid_guards.py --judge llamaguard \
    --input_dir results/preview \
    --files beavertails_llava_cot_tis_clean.json \
            beavertails_llava_cot_tis_gaussian_blur5.json \
            beavertails_llava_cot_tis_gaussian_blur10.json \
    --out_dir results/preview/guard
