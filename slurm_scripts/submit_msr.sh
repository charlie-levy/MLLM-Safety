#!/bin/bash
# Reproduce the MSR-Align row of Table 2 (ORR + FigStep ASR) on LLaVA-CoT.
#   Paper: XSTest=57.6  MMSA=76.2  Avg=66.9  FigStep ASR=23.8
#
# Launches two jobs:
#   1. ORR (XSTest + MMSA)  -> results/orr_base_msr/orr_results.json
#   2. FigStep ASR (clean)  -> results/figstep_noise_sweep/asr_base_msr_clean.json
#
# Usage: bash slurm_scripts/submit_msr.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=job  $2=log  $3=python command
  sbatch --job-name="$1" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --output="logs/$2.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
$3
"
  echo "Submitted: $1"
}

submit "msr_orr"     "msr_orr"     "python code/eval_orr.py --use_msr"
submit "msr_figstep" "msr_figstep" "python code/eval_figstep_clean.py --use_msr"
submit "msr_sqa"     "msr_sqa"     "python code/eval_sqa_noise_sweep.py --use_msr --severity 0 --noise_type gaussian_noise"
