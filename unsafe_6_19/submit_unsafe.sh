#!/bin/bash
# ============================================================================
# submit_unsafe.sh — run the unsafe-images eval (base + TIS, clean/blur20/blur40)
# with the Llama Guard 3 Vision judge. ONE small job on the FREE preemptable
# partition (GPU billed 0.0). Runs preflight FIRST — only submits if it passes.
#
#   bash unsafe_6_19/submit_unsafe.sh 1     # TEST: 1 image per condition
#   bash unsafe_6_19/submit_unsafe.sh       # FULL: all 100 images
#
# The job, in sequence (each model loaded once, freed when its process exits):
#   1. eval base   (clean)
#   2. eval TIS    (clean, blur20, blur40)
#   3. judge ALL with Llama Guard 3 Vision  (ASR = % responses judged UNSAFE)
#   4. combine -> unsafe_6_19/RESULTS_unsafe.json
#
# Everything offline. preemptable can be preempted -> just rerun (still free).
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

LIMIT="${1:-}"                       # "1" => test; empty => full 100
LIMARG=""; [ -n "$LIMIT" ] && LIMARG="--limit $LIMIT"
NAME="unsafe"; [ -n "$LIMIT" ] && NAME="unsafe_test"
TLIM="12:00:00"; [ -n "$LIMIT" ] && TLIM="01:00:00"

# ---- 1) PREFLIGHT on the login node (CPU-only, 0 GPU-hours). Gate the submit. ----
echo "running preflight (CPU, free) ..."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python unsafe_6_19/preflight_unsafe.py $LIMARG \
  || { echo ">>> PREFLIGHT FAILED — NOT submitting. Fix the above first."; exit 1; }

# ---- 2) submit ONE job to the FREE preemptable partition ----
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

# FREE partition (GPU billed 0.0 — does NOT drain the cohort pool).
COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

JID=$(sbatch --parsable $COMMON --job-name="$NAME" --time="$TLIM" \
  --output="logs/${NAME}_%j.log" --wrap="${ENVBLOCK}
python unsafe_6_19/eval_unsafe.py --blur 0 ${LIMARG} || exit 1
python unsafe_6_19/eval_unsafe.py --use_tis --blur 0 20 40 ${LIMARG} || exit 1
python unsafe_6_19/judge_unsafe.py \
  unsafe_6_19/responses_base_clean.json \
  unsafe_6_19/responses_tis_clean.json \
  unsafe_6_19/responses_tis_blur20.json \
  unsafe_6_19/responses_tis_blur40.json || exit 1
python unsafe_6_19/combine_unsafe.py || exit 1")

echo "submitted job $JID  (mode=${LIMIT:+TEST limit=$LIMIT}${LIMIT:-FULL 100})  partition=preemptable (FREE)"
echo "watch:  squeue -u \$USER | grep -E '$NAME'   |   tail -f logs/${NAME}_${JID}.log"
echo "result: unsafe_6_19/RESULTS_unsafe.json"
