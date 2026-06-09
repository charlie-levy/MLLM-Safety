#!/bin/bash
#SBATCH --job-name=figstep_blur
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_blur.log

cd /home/ch169788/llava_cot_eval

# Activate conda
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU

# Install dependencies
pip install -q Pillow

# Set environment
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH

# Run inference with BLUR corruption
echo "=========================================="
echo "  FigStep + gaussian_blur (severity=3)"
echo "=========================================="
python code/run_eval_blur.py
