#!/bin/bash
# Submit clean (no-corruption) FigStep and SQA baselines for BASE and BASE+TIS.
# These provide the sev=0 anchor points for ASR-vs-Utility plots.
# Usage: bash slurm_scripts/submit_clean_baselines.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

for TIS in "" "--use_tis"; do
  TAG=$([ "$TIS" = "--use_tis" ] && echo "tis" || echo "base")

  sbatch --job-name="figstep_clean_${TAG}" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --output="logs/figstep_clean_${TAG}.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/eval_figstep_clean.py $TIS
"
  echo "Submitted: figstep_clean_${TAG}"

  sbatch --job-name="sqa_clean_${TAG}" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=2:00:00 \
         --output="logs/sqa_clean_${TAG}.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/eval_sqa_noise_sweep.py $TIS --severity 0 --noise_type gaussian_noise
"
  echo "Submitted: sqa_clean_${TAG}"
done
