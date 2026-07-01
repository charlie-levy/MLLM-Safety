#!/bin/bash
# Think-in-Safety + random image corruptions — LoRA SFT for LLaVA-CoT — 1x H100.
# Apples-to-apples with finetune_tis.sbatch (only the yaml/dataset differ).
#   sbatch ~/llava_cot_eval/experiments/tis_finetune/finetune_tis_corrupt.sbatch
#SBATCH --job-name=tis_corrupt_lora
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

cd ~/LLaMA-Factory
llamafactory-cli train ~/llava_cot_eval/experiments/tis_finetune/train_llava_cot_tis_corrupt_lora.yaml
