#!/bin/bash
# FigStep noise sweep for SAGE adapter (sev 1-5, gaussian_noise).
# Usage: bash slurm_scripts/submit_sage_figstep_sweep.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

for SEV in 1 2 3 4 5; do
  sbatch --job-name="sage_fig_n${SEV}" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --exclude=evc42 \
         --output="logs/sage_figstep_noise_sev${SEV}.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/eval_figstep_noise_sweep.py --use_sage --severity ${SEV} --noise_type gaussian_noise
"
  echo "Submitted: sage_fig_noise_sev${SEV}"
done
