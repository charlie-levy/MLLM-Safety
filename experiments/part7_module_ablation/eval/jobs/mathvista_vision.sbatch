#!/bin/bash
#SBATCH --job-name=p7_mv_vision
#SBATCH --array=0-3
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ch169788/experiments/part7/logs/%x_%A_%a.out
#SBATCH --error=/home/ch169788/experiments/part7/logs/%x_%A_%a.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part7_module_ablation/eval
python run_mathvista.py --model vision --nshards 4 --shard $SLURM_ARRAY_TASK_ID
