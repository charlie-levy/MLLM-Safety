#!/bin/bash
#SBATCH --job-name=figstep_base
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=figstep_base.log

cd /home/ch169788/llava_cot_eval
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH
python code/eval_figstep_base.py
