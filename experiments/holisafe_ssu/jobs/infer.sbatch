#!/bin/bash
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/ch169788/experiments/holisafe_ssu/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/holisafe_ssu/logs/%x_%j.err

# HoliSafe SI+ST->U inference, all 4 conditions in one model load.
# Usage:  sbatch --job-name=hs_<model> jobs/infer.sbatch <model>
#   model in: base_llama llava_cot llamav_o1 qwen2_5_vl r1_onevision
MODEL="${1:?usage: sbatch --job-name=hs_<model> jobs/infer.sbatch <model>}"

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/holisafe_ssu
python run_holisafe_ssu.py --model "$MODEL" --conditions clean,zoom_blur,snow,glass_blur
