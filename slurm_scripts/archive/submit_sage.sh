#!/bin/bash
# Evaluate SAGE adapter: ORR (XSTest + MMSA), FigStep ASR, and ScienceQA utility.
#
# Launches three jobs:
#   1. ORR        -> results/orr_base_sage/orr_results.json
#   2. FigStep    -> results/figstep_noise_sweep/asr_base_sage_clean.json
#   3. ScienceQA  -> results/sqa_noise_sweep/acc_base_sage_clean.json
#
# Usage: bash slurm_scripts/submit_sage.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=job  $2=log  $3=python command
  sbatch --job-name="$1" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --exclude=evc42 \
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

submit "sage_orr"     "sage_orr"     "python code/eval_orr.py --use_sage"
submit "sage_figstep" "sage_figstep" "python code/eval_figstep_clean.py --use_sage"
submit "sage_sqa"     "sage_sqa"     "python code/eval_sqa_noise_sweep.py --use_sage --severity 0 --noise_type gaussian_noise"
