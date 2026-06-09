#!/bin/bash
# Submit ORR evaluation for BASE and BASE+TIS
# Usage: bash slurm_scripts/submit_orr.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

for TIS in "" "--use_tis"; do
  TAG=$([ "$TIS" = "--use_tis" ] && echo "tis" || echo "base")
  JOB="orr_${TAG}"
  LOG="logs/orr_${TAG}.log"

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
python code/eval_orr.py $TIS
"
  echo "Submitted: $JOB"
done
