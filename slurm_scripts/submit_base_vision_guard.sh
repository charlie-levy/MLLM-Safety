#!/bin/bash
# ============================================================================
# submit_base_vision_guard.sh — re-judge the BASE Llama-3.2-11B-Vision FigStep
# responses with Llama Guard 3 Vision, to get a TRUE ASR (harmful-content %)
# instead of the string-match number (which counts "list not provided / too
# blurry" deflections as attack successes and so massively over-states ASR).
#
# No re-inference: reuses the existing responses_figstep.json in each severity
# dir; judge_figstep_guard.py loads the CLEAN FigStep image + the model response
# and labels safe/unsafe. Output: base_vision_eval/<cond>/asr_guard.json
# (same schema as MSR/VLGuard -> directly comparable).
#
# 6 conditions = sev0..sev5 (clean, blur20, blur40, blur60, blur80, blur100).
#
# Submit from the login node (offline):  bash slurm_scripts/submit_base_vision_guard.sh
# When done:  python code/report_base_vision_guard.py
# ============================================================================
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

for COND in clean blur20 blur40 blur60 blur80 blur100; do
  DIR="results/base_vision_eval/${COND}"
  if [ ! -f "${DIR}/responses_figstep.json" ]; then
    echo "WARN: ${DIR}/responses_figstep.json missing — skipping ${COND}"
    continue
  fi
  JID=$(sbatch --parsable $COMMON --job-name="bvg_${COND}" --time=3:00:00 \
    --output="logs/bvg_${COND}.log" --wrap="${ENVBLOCK}
python code/judge_figstep_guard.py ${DIR}/responses_figstep.json --out_dir ${DIR} || exit 1")
  echo "  bvg_${COND} -> ${JID}"
done

echo ""
echo "Submitted Guard re-scoring for base-vision sev0-5."
echo "Watch:  squeue -u \$USER | grep bvg"
echo "When done:  python code/report_base_vision_guard.py"
