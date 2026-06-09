#!/bin/bash
#SBATCH --job-name=sqa_base
#SBATCH --partition=normal
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=sqa_base.log

cd /home/ch169788/llava_cot_eval
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH
python code/eval_sqa_base.py
