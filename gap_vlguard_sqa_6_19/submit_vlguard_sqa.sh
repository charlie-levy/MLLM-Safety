#!/bin/bash
# ============================================================================
# submit_vlguard_sqa.sh — fill the 4 missing SQA cells of the VLGuard table.
#
#   Existing SQA (LLaVA-1.5-7B, ScienceQA-250 accuracy):  mixed/clean 63.6
#                                                         posthoc/clean 64.4
#   This fills:   mixed   blur20  blur40
#                 posthoc blur20  blur40
#
# Same proven scripts that produced the clean numbers — code/eval_vlguard.py
# (the SQA task now respects --blur_pct; clean behaviour unchanged) + the same
# code/judge_sqa_utility_hf.py (Meta-Llama-3-8B). blur = blur_utils.blur_image.
#
#   bash gap_vlguard_sqa_6_19/submit_vlguard_sqa.sh
#
# 4 jobs, chained one-at-a-time, billed `normal` partition (priority -> not
# preempted), V100-32GB. Each skips itself if its judged JSON already exists.
# These are small (LLaVA-1.5-7B, 250 samples each) so they can run alongside the
# beavertails jobs without much footprint.
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

# login `python` is Python 2 -> use the REU env for the preflight too
set +u
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
set -u

echo "preflight (CPU, free) ..."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python gap_vlguard_sqa_6_19/preflight_vlguard_sqa.py \
  || { echo ">>> PREFLIGHT FAILED — NOT submitting."; exit 1; }

ENVBLOCK='source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval'

COMMON="--partition=normal --gres=gpu:tesla_v100-pcie-32gb:1 --mem=40G --exclude=evc42"

# submit_one <name> <variant> <blur_pct> [dep_jid]
submit_one() {
  local name="$1"; local variant="$2"; local pct="$3"; local dep="${4:-}"
  local d="results/vlguard_eval/${variant}/blur${pct}"
  local jsonl="${d}/raw_vlguard_${variant}_sqa.jsonl"
  local judged="${d}/judged_vlguard_${variant}_sqa.json"
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterany:$dep"
  sbatch --parsable $COMMON $depflag --job-name="$name" --time=02:00:00 \
    --output="logs/${name}_%j.log" --wrap="${ENVBLOCK}
if [ -f ${judged} ]; then echo '[skip] ${judged} exists'; exit 0; fi
python code/eval_vlguard.py --variant ${variant} --task sqa --blur_pct ${pct} || exit 1
python code/judge_sqa_utility_hf.py ${jsonl} || exit 1"
}

J1=$(submit_one vlg_mixed_sqa_blur20   mixed   20)
J2=$(submit_one vlg_mixed_sqa_blur40   mixed   40 "$J1")
J3=$(submit_one vlg_posthoc_sqa_blur20 posthoc 20 "$J2")
J4=$(submit_one vlg_posthoc_sqa_blur40 posthoc 40 "$J3")

echo "submitted 4 jobs (one-at-a-time, normal partition V100-32GB):"
echo "  $J1 vlg_mixed_sqa_blur20    $J2 vlg_mixed_sqa_blur40"
echo "  $J3 vlg_posthoc_sqa_blur20  $J4 vlg_posthoc_sqa_blur40"
echo
echo "watch:  squeue -u \$USER"
echo "table:  python gap_vlguard_sqa_6_19/report_vlguard_sqa.py   (any time)"
