#!/bin/bash
# ============================================================================
# Baseline eval: meta-llama/Llama-3.2-11B-Vision-Instruct (no safety, no CoT)
# FigStep ASR + XSTest/MMSA ORR + ScienceQA utility for ONE blur level.
# All metrics string-matching, no judge jobs.
#
# SAFETY MODEL (read this):
#   * Runs ONE condition per call — no 6-way / 57-way fan-out.
#   * Submits to the FREE 'preemptable' partition (GPU billed 0.0 -> does NOT
#     draw down the course GPU-hour pool). Jobs CAN be preempted; if so, just
#     re-run the same command (free).
#   * A CPU-only preflight (code/preflight.py) must PASS before anything is
#     submitted — catches wrong dataset counts / missing weights for 0 GPU cost.
#   * Tight --time so a hung job dies fast instead of sitting on a node.
#
#   conda activate REU   # preflight needs pandas/PIL from the env
#   bash slurm_scripts/submit_base_vision.sh 0      # clean
#   bash slurm_scripts/submit_base_vision.sh 20     # blur20
#   ... one at a time: 0 20 40 60 80 100
# When all six are done:  python code/report_base_vision.py results/base_vision_eval
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

PCT="${1:-}"
if [ -z "$PCT" ]; then
  echo "Usage: bash slurm_scripts/submit_base_vision.sh <blur_pct: 0|20|40|60|80|100>"
  echo "Runs ONE condition on the FREE preemptable partition. Run them one at a time."
  exit 1
fi
case "$PCT" in
  0|20|40|60|80|100) ;;
  *) echo "ERROR: blur_pct must be one of 0 20 40 60 80 100 (got '$PCT')"; exit 1 ;;
esac
if [ "$PCT" -eq 0 ]; then COND="clean"; else COND="blur${PCT}"; fi

# ---- Preflight: CPU only, 0 GPU-hours. No submit unless it passes. ----
echo "Running preflight (CPU, 0 GPU-hours) ..."
if ! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python code/preflight.py --eval base_vision --blur_pct "$PCT"; then
  echo ""
  echo "PREFLIGHT FAILED — nothing submitted. Fix the issues above and retry."
  exit 1
fi

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

# FREE partition (GPU billed 0.0). H100 = fastest = least time exposed to preemption.
# Still exclude evc42 (bad node). Tight 4h cap for a ~2h job.
COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

JID=$(sbatch --parsable $COMMON --job-name="bv_${COND}" --time=04:00:00 \
  --output="logs/bv_${COND}.log" --wrap="${ENVBLOCK}
python code/eval_base_vision.py --blur_pct ${PCT} || exit 1")

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Submitted bv_${COND}: job=${JID}"
echo "  partition=preemptable (FREE, billed 0.0)  ·  time cap 4h"
echo "  Watch:    squeue -u \$USER | grep bv_${COND}"
echo "  Log:      tail -f logs/bv_${COND}.log"
echo "  Preempted? Re-run:  bash slurm_scripts/submit_base_vision.sh ${PCT}"
echo "──────────────────────────────────────────────────────────────"
