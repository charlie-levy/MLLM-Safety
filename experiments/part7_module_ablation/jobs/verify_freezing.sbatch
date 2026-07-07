#!/bin/bash
# Part 7 module-ablation — PROVE freezing for all 3 configs before training.
# Loads the 11B model 3x through LLaMA-Factory's own pipeline and checks trainable params.
#   sbatch ~/llava_cot_eval/experiments/part7_module_ablation/jobs/verify_freezing.sbatch
#SBATCH --job-name=p7_verify_freeze
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ch169788/experiments/part7/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part7/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

cd ~/LLaMA-Factory                      # so `dataset_dir: data` in the yaml resolves
P=~/llava_cot_eval/experiments/part7_module_ablation
set -x
echo "############## #1 LLM-only (expect language_model) ##############"
python $P/verify_freezing.py $P/configs/train_zoomblur75_llm_lora.yaml    --expect language_model
echo "############## #2 Vision-only (expect vision_model) ##############"
python $P/verify_freezing.py $P/configs/train_zoomblur75_vision_lora.yaml --expect vision_model
echo "############## #3 Both (expect language_model + vision_model) ##############"
python $P/verify_freezing.py $P/configs/train_zoomblur75_both_lora.yaml   --expect both
