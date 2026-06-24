#!/bin/bash
#SBATCH --job-name=hhc_candidates
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.log

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval

# stage the model to node-local SSD for a ~4s load instead of ~10 min off Lustre
mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--Xkev--Llama-3.2V-11B-cot /tmp/hf/hub/ 2>/dev/null
export HF_HOME=/tmp/hf

# graded SMALL corruption ladder on the 15 harmful_health_content drug candidates.
# Find where TIS is safe (refuses) and where it first breaks (complies). No blur-20.
python code/run_candidates.py --model llava_cot_tis \
    --items datasets/new_attacks/hhc/hhc.json \
    --sweep none:0,gaussian_blur:5,gaussian_blur:10,pixelate:10,pixelate:20 \
    --out results/candidates/hhc