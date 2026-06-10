#!/bin/bash
# Find a real FigStep image where TIS refuses clean but complies after corruption.
# Faithful to the ASR eval (uses load_figstep). Default = blur sev4 (highest TIS bypass).
# Usage: bash slurm_scripts/submit_find_bypass.sh [noise_type] [severity]

cd /home/ch169788/llava_cot_eval
mkdir -p logs

NOISE="${1:-gaussian_blur}"
SEV="${2:-4}"

sbatch --job-name="find_byp" \
       --partition=normal \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=64G \
       --time=4:00:00 \
       --output="logs/find_bypass.log" \
       --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/find_bypass_image.py --noise_type $NOISE --severity $SEV
"
echo "Submitted find_bypass: $NOISE sev$SEV"
