#!/bin/bash
# pull_results.sh — Pull all Newton results locally, organized.
# Run on your Mac AFTER `squeue -u ch169788` is empty.
# Sorts noise vs blur into the right folders to match the repo convention.
#
# Usage: bash pull_results.sh

set -u
NEWTON="ch169788@newton.ist.ucf.edu:/home/ch169788/llava_cot_eval/results"
DEST=~/Desktop/REU/llava_cot_eval/results_newton

mkdir -p "$DEST/orr" "$DEST/orr_noise_sweep" "$DEST/orr_blur_sweep" \
         "$DEST/sqa_noise_sweep" "$DEST/sqa_blur_sweep" \
         "$DEST/figstep_noise_sweep" "$DEST/figstep_blur_sweep"

echo "==> ORR noise sweep (noise only)"
scp "$NEWTON/orr_noise_sweep/*gaussian_noise*.json" "$DEST/orr_noise_sweep/"
scp "$NEWTON/orr_noise_sweep/*gaussian_noise*.csv"  "$DEST/orr_noise_sweep/" 2>/dev/null || true

echo "==> ORR blur sweep (blur only)"
scp "$NEWTON/orr_noise_sweep/*gaussian_blur*.json" "$DEST/orr_blur_sweep/"
scp "$NEWTON/orr_noise_sweep/*gaussian_blur*.csv"  "$DEST/orr_blur_sweep/" 2>/dev/null || true

echo "==> MSR-Align ORR"
scp "$NEWTON/orr_base_msr/orr_results.json" "$DEST/orr/orr_base_msr.json"

echo "==> MSR-Align FigStep ASR + CSV"
scp "$NEWTON/figstep_noise_sweep/asr_base_msr_clean.json"       "$DEST/figstep_noise_sweep/"
scp "$NEWTON/figstep_noise_sweep/responses_base_msr_clean.csv"  "$DEST/figstep_noise_sweep/" 2>/dev/null || true

echo "==> MSR-Align SQA (raw + regex)"
scp "$NEWTON/sqa_noise_sweep/raw_base_msr_clean.jsonl" "$DEST/sqa_noise_sweep/" 2>/dev/null || echo "  (not yet available)"
scp "$NEWTON/sqa_noise_sweep/acc_base_msr_clean.json"  "$DEST/sqa_noise_sweep/" 2>/dev/null || echo "  (not yet available)"

echo "==> SQA raw responses + regex acc + CSV (66 files)"
scp "$NEWTON/sqa_noise_sweep/raw_*.jsonl"       "$DEST/sqa_noise_sweep/"
scp "$NEWTON/sqa_noise_sweep/acc_*.json"        "$DEST/sqa_noise_sweep/"
scp "$NEWTON/sqa_noise_sweep/responses_*.csv"   "$DEST/sqa_noise_sweep/"

echo "==> Sorting blur files into sqa_blur_sweep/"
mv "$DEST"/sqa_noise_sweep/*blur* "$DEST/sqa_blur_sweep/" 2>/dev/null || true

echo ""
echo "Done pulling. Next: run the LLaMA-3 utility judge locally:"
echo "  python3 code/judge_sqa_utility.py --dir results_newton/sqa_noise_sweep"
echo "  python3 code/judge_sqa_utility.py --dir results_newton/sqa_blur_sweep"
