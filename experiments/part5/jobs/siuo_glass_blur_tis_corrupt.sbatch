#!/bin/bash
# Part 5: SIUO + glass_blur(sev5) on the CORRUPT-TIS checkpoint (tis_corrupt). Responses only, no judge.
# Submit ONLY after training job 677046 is COMPLETED and saves/llava_cot_tis_corrupt_lora/ has an adapter.
#SBATCH --job-name=p5_glass_blur_tis_corrupt
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ch169788/experiments/part5/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part5/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part5
python run_inference.py --model tis_corrupt --condition glass_blur \
    --output_dir /home/ch169788/experiments/part5/results
