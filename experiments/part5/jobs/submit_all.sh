#!/bin/bash
# Part 5: submit all 4 SIUO corruption-eval jobs (zoom_blur + glass_blur x tis_lora + tis_corrupt).
# Responses only, no judge. Run only after BOTH training jobs (677046, 677047) are COMPLETED
# and both adapters exist. To launch per-checkpoint instead, use submit_tis_lora.sh /
# submit_tis_corrupt.sh as each training run finishes.
set -e
mkdir -p /home/ch169788/experiments/part5/results /home/ch169788/experiments/part5/logs
cd ~/llava_cot_eval/experiments/part5/jobs
sbatch siuo_zoom_blur_tis_lora.sbatch
sbatch siuo_glass_blur_tis_lora.sbatch
sbatch siuo_zoom_blur_tis_corrupt.sbatch
sbatch siuo_glass_blur_tis_corrupt.sbatch
