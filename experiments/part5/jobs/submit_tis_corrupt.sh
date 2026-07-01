#!/bin/bash
# Part 5: submit the 2 SIUO corruption-eval jobs for the CORRUPT-TIS checkpoint (tis_corrupt).
# Run once training job 677046 is COMPLETED and saves/llava_cot_tis_corrupt_lora/ has an adapter.
set -e
mkdir -p /home/ch169788/experiments/part5/results /home/ch169788/experiments/part5/logs
cd ~/llava_cot_eval/experiments/part5/jobs
sbatch siuo_zoom_blur_tis_corrupt.sbatch
sbatch siuo_glass_blur_tis_corrupt.sbatch
