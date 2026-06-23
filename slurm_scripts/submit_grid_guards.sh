#!/bin/bash
# ============================================================================
# Apples-to-apples ASR on model_responses/ with TWO text-only judges:
#   * LLaMA-Guard 3 Vision (text-only)  meta-llama/Llama-Guard-3-11B-Vision
#   * WildGuard                         allenai/wildguard
#
# Both judges see the same (prompt, response) text -> directly comparable ASR.
#
# DAG (slurm afterok):
#     preflight (8 samples/file, BOTH models) ──> full (all 12 cells) ──> report
# The preflight loads both guards and judges 8 real samples per file (~5 min),
# so a model-load / chat-template / parse bug fails cheap before the full run.
#
# Everything runs OFFLINE on ONE preemptable H100 (billed 0.0). Submit from login:
#     bash slurm_scripts/submit_grid_guards.sh
#
# Prereq: WildGuard cached ->  hf download allenai/wildguard
#         (Llama-Guard-3-11B-Vision is already cached.)
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs results/grid_guard_eval

# Sanity: refuse to submit if the grid or the WildGuard weights aren't present.
N_JSON=$(ls model_responses/*.json 2>/dev/null | wc -l)
if [ "$N_JSON" -eq 0 ]; then
  echo "ERROR: no model_responses/*.json found in $REPO (did you 'git pull'?)" >&2; exit 1
fi
if [ -z "$(ls -d ~/.cache/huggingface/hub/models--allenai--wildguard/snapshots/* 2>/dev/null)" ]; then
  echo "ERROR: WildGuard not cached. Run on the login node:  hf download allenai/wildguard" >&2; exit 1
fi
echo "Found $N_JSON response files + WildGuard weights. Submitting..."

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

COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

submit() {  # submit <jobname> <timelimit> <dependency-or-empty> <python-command>
  local name="$1" tlimit="$2" dep="$3" cmd="$4"
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" $depflag \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

PF=$(submit grid_guard_preflight 0:30:00 "" \
  "python code/judge_grid_guards.py --judge both --limit 8 --out_dir results/grid_guard_eval/_preflight")

FULL=$(submit grid_guard_full 6:00:00 "$PF" \
  "python code/judge_grid_guards.py --judge both --out_dir results/grid_guard_eval")

REP=$(submit grid_guard_report 0:20:00 "$FULL" \
  "python code/report_grid_guards.py results/grid_guard_eval")

echo "──────────────────────────────────────────────────────────────"
echo "preflight=$PF  ->  full=$FULL  ->  report=$REP"
echo "Watch:   squeue -u \$USER | grep grid_guard"
echo "Preflight log:  tail -f logs/grid_guard_preflight.log"
echo "Final table:    cat logs/grid_guard_report.log   (also results/grid_guard_eval/asr_summary.json)"
echo "──────────────────────────────────────────────────────────────"