#!/bin/bash
#SBATCH --job-name=p10_vls_bench
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ch169788/experiments/part10/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part10/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part10_paper_judges
python run_inference.py --dataset vls_bench --conditions clean,zoom_blur
