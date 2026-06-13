#!/bin/bash
# pull_results.sh — Pull ALL Newton eval outputs to the Mac in ONE ssh session.
# Run ON YOUR MAC:  bash scripts/pull_results.sh
#
# Uses rsync (one connection = ONE password prompt, incremental on re-runs) and
# grabs EVERY result file we care about, across all corruption families
# (noise / blur / jpeg / brightness / pixelate) and the clean baselines:
#   *.json          metric files (asr_*, orr_*, judged_*, acc_*)
#   *.csv           per-sample FULL model responses (responses_*.csv) + combined CSV
#   raw_*.jsonl     raw SQA generations (the unjudged model outputs)
#   *.png           example image strips (noise_examples/, corruption_examples/)
# Nothing else is transferred (no checkpoints), so it stays small + fast.

set -e
NEWTON="ch169788@newton.ist.ucf.edu"
REMOTE="/home/ch169788/llava_cot_eval/results"
LOCAL="results_newton"
mkdir -p "$LOCAL"

echo "=== Pulling all results from Newton (one ssh session) ==="
rsync -avm \
  --include='*/' \
  --include='*.json' \
  --include='*.csv' \
  --include='raw_*.jsonl' \
  --include='*.png' \
  --exclude='*' \
  "$NEWTON:$REMOTE/" "$LOCAL/"

echo ""
echo "Pull complete. Now regenerate everything:"
echo "  python3 code/safety_analysis.py     # validated analysis + combo charts"
echo "  python3 code/plot_results.py         # legacy per-model bar charts"
