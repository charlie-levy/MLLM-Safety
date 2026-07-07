#!/bin/bash
# Part 7 — build offline SQA-test + MathVista-testmini files on a COMPUTE node.
# The parquets are already in the HF cache (downloaded on the login node), so this runs
# OFFLINE and just does the parquet->arrow generation + image dumping, where the compute
# node's high process limit avoids the login-node RLIMIT_NPROC crash.
#   sbatch ~/llava_cot_eval/experiments/part7_module_ablation/eval/jobs/prep_data.sbatch
#SBATCH --job-name=p7_prep_data
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ch169788/experiments/part7/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part7/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

E=~/llava_cot_eval/experiments/part7_module_ablation/eval
echo "############## ScienceQA test ##############"
python3 $E/prep_sqa_test.py
echo "############## MathVista testmini ##############"
python3 $E/prep_mathvista.py