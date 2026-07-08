#!/bin/bash
# Part 7 — SQA prompt PILOT: verify the answer-forcing prompt makes llm emit "The answer is X."
# before committing all 3 models to a full re-run. Runs 1/100 of the test set (~42 items) into a
# throwaway dir and prints extraction_failed. Baseline (old prompt) was ~15% failed; want ~0.
#   sbatch ~/llava_cot_eval/experiments/part7_module_ablation/eval/jobs/sqa_pilot.sbatch
#SBATCH --job-name=p7_sqa_pilot
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=/home/ch169788/experiments/part7/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part7/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part7_module_ablation/eval
python run_sqa.py --model llm --nshards 100 --shard 0 --output_dir /home/ch169788/experiments/part7/sqa_pilot
echo "===== extraction check on the pilot (want extraction_failed ~0) ====="
python score_sqa.py /home/ch169788/experiments/part7/sqa_pilot/shard_llm_0of100.jsonl
