#!/bin/bash
# Part 6: submit all 4 SIUO corruption-eval jobs (zoom_blur sev2 + glass_blur sev3 x tis_lora + tis_corrupt).
# Responses only, no judge. Re-run of Part 5 at LOWER severities; both adapters already exist
# (saves/llava_cot_tis_lora, saves/llava_cot_tis_corrupt_lora). To launch per-checkpoint instead,
# use submit_tis_lora.sh / submit_tis_corrupt.sh.
set -e
mkdir -p /home/ch169788/experiments/part6/results /home/ch169788/experiments/part6/logs
cd ~/llava_cot_eval/experiments/part6/jobs
sbatch siuo_zoom_blur_tis_lora.sbatch
sbatch siuo_glass_blur_tis_lora.sbatch
sbatch siuo_zoom_blur_tis_corrupt.sbatch
sbatch siuo_glass_blur_tis_corrupt.sbatch
