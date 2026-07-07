#!/bin/bash
# Part 7 module-ablation — training #3: LoRA on BOTH the LLM and the Vision Encoder, TIS+75% zoom_blur(sev2).
#   sbatch ~/llava_cot_eval/experiments/part7_module_ablation/jobs/finetune_both.sbatch
#SBATCH --job-name=p7_both_lora
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ch169788/experiments/part7/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part7/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/home/ch169788/experiments/part7/hfcache/both   # per-job cache -> no concurrent-build race

cd ~/LLaMA-Factory
llamafactory-cli train ~/llava_cot_eval/experiments/part7_module_ablation/configs/train_zoomblur75_both_lora.yaml
