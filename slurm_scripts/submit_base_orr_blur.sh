#!/bin/bash
# Base model (no adapter) ORR at blur20 and blur40 — the only gap not covered
# by submit_all_missing.sh.
# Output: results/orr_blur_pct/orr_base_gaussian_blur_pct_p{20,40}.json
#
#   bash slurm_scripts/submit_base_orr_blur.sh
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

ENVBLOCK='source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval'

COMMON="--partition=normal --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

for P in 20 40; do
  JID=$(sbatch --parsable $COMMON --job-name="base_orr_p${P}" --time=6:00:00 \
    --output="logs/base_orr_p${P}.log" --wrap="${ENVBLOCK}
python code/eval_orr_noise_sweep.py --blur_pct ${P} || exit 1")
  echo "base_orr_p${P} -> ${JID}"
done

echo "2 jobs submitted."
