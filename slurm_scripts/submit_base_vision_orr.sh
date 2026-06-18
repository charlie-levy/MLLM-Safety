#!/bin/bash
# ============================================================================
# submit_base_vision_orr.sh — re-judge the BASE Llama-3.2-11B-Vision ORR with the
# LLaMA-3-8B judge (the SAME judge as MSR/VLGuard), replacing the string-match ORR.
#
# Why: eval_base_vision.py scores XSTest ORR with is_refusal() but MMSA ORR with
# is_mmsa_over_refusal() (only "No" answers). Under blur the model says
# "I cannot determine — too blurry", which counts for XSTest (->100%) but not MMSA
# (->~7%), so the two columns diverge and the MMSA ORR is badly understated.
#
# No re-inference: builds responses_orr.csv from the existing responses_xstest.json
# + responses_mmsa.json, then runs judge_safety_hf --mode orr + format_orr_results.
# Output per cond: judged_llama_orr.json (xstest/mmsa/avg orr_pct), comparable to
# MSR/VLGuard.
#
# Submit from the login node (offline):  bash slurm_scripts/submit_base_vision_orr.sh
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

# FREE preemptable partition (GPU billed 0.0 — does NOT drain the cohort pool).
COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

for COND in clean blur20 blur40 blur60 blur80 blur100; do
  DIR="results/base_vision_eval/${COND}"
  if [ ! -f "${DIR}/responses_xstest.json" ] || [ ! -f "${DIR}/responses_mmsa.json" ]; then
    echo "WARN: ${DIR} missing xstest/mmsa responses — skipping ${COND}"
    continue
  fi
  JID=$(sbatch --parsable $COMMON --job-name="bvo_${COND}" --time=4:00:00 \
    --output="logs/bvo_${COND}.log" --wrap="${ENVBLOCK}
python code/build_orr_csv.py ${DIR} \
 && python code/judge_safety_hf.py --mode orr ${DIR}/responses_orr.csv \
 && python code/format_orr_results.py ${DIR} || exit 1")
  echo "  bvo_${COND} -> ${JID}"
done

echo ""
echo "Submitted LLaMA-3 ORR re-judging for base-vision sev0-5."
echo "Watch:  squeue -u \$USER | grep bvo"
echo "When done:  python code/report_base_vision_guard.py"
