#!/bin/bash
# ============================================================================
# submit_unsafe.sh — run the unsafe-images ASR experiment.
#   base(clean) + TIS(clean, blur_20, blur_40)  ->  Llama Guard 3 Vision judge
#   ASR = % responses judged UNSAFE (refusal / "couldn't tell" = NOT a success).
#
#   bash unsafe_6_19/submit_unsafe.sh pilot   # PILOT: image 0 only, 4 conditions
#   bash unsafe_6_19/submit_unsafe.sh         # FULL: all 100 images
#
# ONE small job on the FREE preemptable partition (GPU billed 0.0). Runs preflight
# (CPU, free) FIRST and only submits if it passes. eval + judge are INCREMENTAL +
# RESUMABLE, so if the job is preempted you just re-run this and it continues.
#
# Pipeline inside the job (each model loaded once; freed when its process exits):
#   eval base (clean) -> eval TIS (clean,20,40) -> judge ALL -> build_results -> calc_asr
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

MODE="${1:-full}"
if [ "$MODE" = "pilot" ]; then
  PFLAG="--pilot"; SUF="_pilot"; NAME="unsafe_pilot"; TLIM="01:00:00"; PRE="--limit 1"
else
  PFLAG="";        SUF="";       NAME="unsafe_full";  TLIM="12:00:00"; PRE=""
fi

# ---- 1) PREFLIGHT on the login node (CPU-only, 0 GPU-hours). Gate the submit. ----
echo "preflight (CPU, free) ..."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python unsafe_6_19/preflight_unsafe.py $PRE \
  || { echo ">>> PREFLIGHT FAILED — NOT submitting."; exit 1; }

# ---- 2) submit ONE job to the FREE preemptable partition ----
ENVBLOCK='source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/unsafe_6_19:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval'

# FREE partition (GPU billed 0.0 — does NOT drain the cohort pool).
COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=40G --exclude=evc42"

JID=$(sbatch --parsable $COMMON --job-name="$NAME" --time="$TLIM" \
  --output="logs/${NAME}_%j.log" --wrap="${ENVBLOCK}
python unsafe_6_19/eval_unsafe.py --blur 0 ${PFLAG} || exit 1
python unsafe_6_19/eval_unsafe.py --use_tis --blur 0 20 40 ${PFLAG} || exit 1
python unsafe_6_19/judge_unsafe.py \
  unsafe_6_19/responses_base_clean${SUF}.json \
  unsafe_6_19/responses_tis_clean${SUF}.json \
  unsafe_6_19/responses_tis_blur_20${SUF}.json \
  unsafe_6_19/responses_tis_blur_40${SUF}.json || exit 1
python unsafe_6_19/build_results.py ${PFLAG} || exit 1
python unsafe_6_19/calc_asr.py ${PFLAG} || exit 1")

echo "submitted job $JID  (mode=$MODE)  partition=preemptable (FREE)"
echo "watch:   squeue -u \$USER | grep $NAME    |    tail -f logs/${NAME}_${JID}.log"
if [ "$MODE" = "pilot" ]; then
  echo "result:  unsafe_6_19/pilot_output.json   (READ all 4 + the ASR table before the full run)"
else
  echo "result:  unsafe_6_19/unsafe100_asr_results.json  +  calc_asr table"
fi
