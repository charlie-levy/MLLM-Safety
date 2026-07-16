#!/bin/bash
#SBATCH --job-name=p11_zoomblur
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ch169788/experiments/part11/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part11/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part11_thinking_budget
python run_budget_pilot.py --condition zoom_blur --severity 2 --n_samples 50 --budgets 0,512,2048,natural
