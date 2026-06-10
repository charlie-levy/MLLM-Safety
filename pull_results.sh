#!/bin/bash
# pull_results.sh — Pull all Newton results locally, organized.
# Run on your Mac AFTER `squeue -u ch169788` is empty.
# Sorts noise vs blur into the right folders to match the repo convention.
#
# Usage: bash pull_results.sh

set -u
NEWTON="ch169788@newton.ist.ucf.edu:/home/ch169788/llava_cot_eval/results"
DEST=~/Desktop/REU/llava_cot_eval/results_newton

mkdir -p "$DEST/orr" "$DEST/orr_noise_sweep" \
         "$DEST/sqa_noise_sweep" "$DEST/sqa_blur_sweep" \
         "$DEST/figstep_noise_sweep" "$DEST/figstep_blur_sweep"

echo "==> ORR noise+blur sweep (20 files)"
scp "$NEWTON/orr_noise_sweep/*.json" "$DEST/orr_noise_sweep/"

echo "==> MSR-Align ORR"
scp "$NEWTON/orr_base_msr/orr_results.json" "$DEST/orr/orr_base_msr.json"

echo "==> MSR-Align FigStep ASR"
scp "$NEWTON/figstep_noise_sweep/asr_base_msr_clean.json" "$DEST/figstep_noise_sweep/"

echo "==> SQA raw responses + regex acc (44 files)"
scp "$NEWTON/sqa_noise_sweep/raw_*.jsonl" "$DEST/sqa_noise_sweep/"
scp "$NEWTON/sqa_noise_sweep/acc_*.json"  "$DEST/sqa_noise_sweep/"

echo "==> Sorting blur files into sqa_blur_sweep/"
mv "$DEST"/sqa_noise_sweep/*blur* "$DEST/sqa_blur_sweep/" 2>/dev/null || true

echo ""
echo "Done pulling. Next: run the LLaMA-3 utility judge locally:"
echo "  python3 code/judge_sqa_utility.py --dir results_newton/sqa_noise_sweep"
echo "  python3 code/judge_sqa_utility.py --dir results_newton/sqa_blur_sweep"
