#!/bin/bash
#SBATCH --job-name=sqa_blur
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_sqa_blur.log

cd /home/ch169788/llava_cot_eval
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
pip install -q Pillow
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH

python code/eval_scienceqa_blur.py
