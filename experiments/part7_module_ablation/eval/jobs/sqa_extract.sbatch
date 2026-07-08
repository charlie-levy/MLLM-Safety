#!/bin/bash
# Part 7 — SQA hybrid answer extraction (ScienceQA rule first, MathVista-style Llama-3-8B
# fallback for the ~15% that omit the letter). Runs over the merged raw_{llm,vision,both}.jsonl.
#   sbatch ~/llava_cot_eval/experiments/part7_module_ablation/eval/jobs/sqa_extract.sbatch
#SBATCH --job-name=p7_sqa_extract
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ch169788/experiments/part7/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part7/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part7_module_ablation/eval
python extract_sqa.py --dir /home/ch169788/experiments/part7/sqa
