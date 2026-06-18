#!/bin/bash
# Judge the SQA raw responses from the realistic-corruption sweeps
# (results/sqa_jpeg_pct, results/sqa_brightness_pct, results/sqa_pixelate_pct).
# One judge job per corruption (parallel). HF offline — the 8B judge model is
# already cached (the noise/blur judge runs succeeded).
# Usage: bash slurm_scripts/submit_judge_realistic.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

for CORR in jpeg brightness pixelate; do
  sbatch --job-name="judge_${CORR}" \
         --partition=preemptable --qos=preemptable \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --exclude=evc42 \
         --output="logs/judge_${CORR}_pct.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ch169788/llava_cot_eval
python code/judge_sqa_utility_hf.py --dir results/sqa_${CORR}_pct --skip-existing
"
  echo "Submitted: judge_${CORR}"
done
