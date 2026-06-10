#!/bin/bash
#SBATCH --job-name=sqa_noise
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_sqa_noise.log

cd /home/ch169788/llava_cot_eval
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
pip install -q Pillow
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH

python code/eval_scienceqa_noise.py
