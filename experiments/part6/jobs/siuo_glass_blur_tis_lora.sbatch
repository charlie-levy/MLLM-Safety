#!/bin/bash
# Part 6: SIUO + glass_blur(sev3) on the CLEAN-TIS checkpoint (tis_lora). Responses only, no judge.
# Re-run of Part 5 at lower severity. Adapter already exists (saves/llava_cot_tis_lora).
#SBATCH --job-name=p6_glass_blur_tis_lora
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ch169788/experiments/part6/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part6/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part6
python run_inference.py --model tis_lora --condition glass_blur \
    --output_dir /home/ch169788/experiments/part6/results
