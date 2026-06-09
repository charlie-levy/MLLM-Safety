#!/bin/bash
# Submit FigStep blur sweep: severity 1-5 x base/base+tis
# Usage: bash slurm_scripts/submit_figstep_blur_sweep.sh

cd /home/ch169788/llava_cot_eval

for SEV in 1 2 3 4 5; do
  for TIS in "" "--use_tis"; do
    TAG=$([ "$TIS" = "--use_tis" ] && echo "tis" || echo "base")
    JOB="figstep_blur_${TAG}_sev${SEV}"
    LOG="logs/figstep_blur_${TAG}_sev${SEV}.log"

    sbatch --job-name="$JOB" \
           --partition=normal \
           --gres=gpu:tesla_v100-pcie-32gb:1 \
           --mem=64G \
           --time=4:00:00 \
           --output="$LOG" \
           --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/eval_figstep_noise_sweep.py $TIS --severity $SEV --noise_type gaussian_blur
"
    echo "Submitted: $JOB"
  done
done
