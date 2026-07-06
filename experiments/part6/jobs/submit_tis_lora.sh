#!/bin/bash
# Part 6: submit the 2 SIUO corruption-eval jobs for the CLEAN-TIS checkpoint (tis_lora).
# zoom_blur sev2 + glass_blur sev3. Adapter saves/llava_cot_tis_lora already exists.
set -e
mkdir -p /home/ch169788/experiments/part6/results /home/ch169788/experiments/part6/logs
cd ~/llava_cot_eval/experiments/part6/jobs
sbatch siuo_zoom_blur_tis_lora.sbatch
sbatch siuo_glass_blur_tis_lora.sbatch
