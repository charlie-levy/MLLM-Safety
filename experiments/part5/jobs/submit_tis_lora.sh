#!/bin/bash
# Part 5: submit the 2 SIUO corruption-eval jobs for the CLEAN-TIS checkpoint (tis_lora).
# Run once training job 677047 is COMPLETED and saves/llava_cot_tis_lora/ has an adapter.
set -e
mkdir -p /home/ch169788/experiments/part5/results /home/ch169788/experiments/part5/logs
cd ~/llava_cot_eval/experiments/part5/jobs
sbatch siuo_zoom_blur_tis_lora.sbatch
sbatch siuo_glass_blur_tis_lora.sbatch
