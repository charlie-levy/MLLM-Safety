#!/bin/bash
# Submit SQA utility noise sweep: severity 1-5 x base/base+tis x noise/blur
# Usage: bash slurm_scripts/submit_sqa_noise_sweep.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

for NOISE in gaussian_noise gaussian_blur; do
  for SEV in 1 2 3 4 5; do
    for TIS in "" "--use_tis"; do
      TAG=$([ "$TIS" = "--use_tis" ] && echo "tis" || echo "base")
      SHORT=$([ "$NOISE" = "gaussian_noise" ] && echo "noise" || echo "blur")
      JOB="sqa_${SHORT}_${TAG}_s${SEV}"
      LOG="logs/sqa_${SHORT}_${TAG}_sev${SEV}.log"

      sbatch --job-name="$JOB" \
             --partition=normal \
             --gres=gpu:1 \
             --mem=64G \
             --time=4:00:00 \
             --output="$LOG" \
             --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/eval_sqa_noise_sweep.py $TIS --severity $SEV --noise_type $NOISE
"
      echo "Submitted: $JOB"
    done
  done
done
