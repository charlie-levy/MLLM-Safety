#!/bin/bash
# pull_results.sh — Pull all Newton eval results to Mac results_newton/.
# Run ON YOUR MAC: bash scripts/pull_results.sh
#
# Pulls:
#   noise-% results (figstep / orr / sqa)  -> results_newton/*_noise_pct/
#   blur-%  results (figstep / orr / sqa)  -> results_newton/*_blur_pct/
#   example image strips                   -> results_newton/noise_examples/
#   combined FigStep TIS responses CSV     -> results_newton/

set -e
NEWTON="ch169788@newton.ist.ucf.edu"
REMOTE="/home/ch169788/llava_cot_eval/results"
LOCAL="results_newton"

echo "=== Pulling noise-% results ==="
mkdir -p "$LOCAL/figstep_noise_pct" "$LOCAL/orr_noise_pct" "$LOCAL/sqa_noise_pct"
scp "$NEWTON:$REMOTE/figstep_noise_pct/*.json" "$LOCAL/figstep_noise_pct/" 2>/dev/null || true
scp "$NEWTON:$REMOTE/orr_noise_pct/*.json"     "$LOCAL/orr_noise_pct/"     2>/dev/null || true
scp "$NEWTON:$REMOTE/sqa_noise_pct/judged_*.json" "$LOCAL/sqa_noise_pct/" 2>/dev/null || true

echo "=== Pulling blur-% results ==="
mkdir -p "$LOCAL/figstep_blur_pct" "$LOCAL/orr_blur_pct" "$LOCAL/sqa_blur_pct"
scp "$NEWTON:$REMOTE/figstep_blur_pct/*.json" "$LOCAL/figstep_blur_pct/" 2>/dev/null || true
scp "$NEWTON:$REMOTE/orr_blur_pct/*.json"     "$LOCAL/orr_blur_pct/"     2>/dev/null || true
scp "$NEWTON:$REMOTE/sqa_blur_pct/judged_*.json" "$LOCAL/sqa_blur_pct/" 2>/dev/null || true

echo "=== Pulling example strips ==="
mkdir -p "$LOCAL/noise_examples"
scp "$NEWTON:$REMOTE/noise_examples/*.png" "$LOCAL/noise_examples/" 2>/dev/null || true

echo "=== Pulling combined FigStep TIS responses ==="
scp "$NEWTON:$REMOTE/figstep_tis_all_responses.csv" "$LOCAL/" 2>/dev/null || true

echo ""
echo "Pull complete. Now regenerate all plots:"
echo "  python3 code/plot_results.py"
