#!/bin/bash
# Part 7 — build offline SQA-test + MathVista-testmini files on a COMPUTE node.
# The login node's process cap (RLIMIT_NPROC) is saturated and can't even create pyarrow's
# thread pool; a compute node has a high process limit, so pyarrow reads the (local, already-
# cached) parquet files fine. No network/hub needed — the prep scripts read parquet directly.
#   sbatch ~/llava_cot_eval/experiments/part7_module_ablation/eval/jobs/prep_data.sbatch
#SBATCH --job-name=p7_prep_data
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ch169788/experiments/part7/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part7/logs/%x_%j.err

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU

E=~/llava_cot_eval/experiments/part7_module_ablation/eval
echo "############## ScienceQA test ##############"
python3 $E/prep_sqa_test.py
echo "############## MathVista testmini ##############"
python3 $E/prep_mathvista.py