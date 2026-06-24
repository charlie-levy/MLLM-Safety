#!/bin/bash
#SBATCH --job-name=figstep_noise
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.log

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval

# stage TIS + Llama-Guard to node-local SSD for fast loads
mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--Xkev--Llama-3.2V-11B-cot /tmp/hf/hub/ 2>/dev/null
cp -an ~/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-11B-Vision /tmp/hf/hub/ 2>/dev/null
export HF_HOME=/tmp/hf

# ---- INFERENCE: noise sweep, apples-to-apple FigStep vs BeaverTails, TIS, n=10 ----
# FigStep = harm in the IMAGE (typographic) -> expect clean-refuse -> noise-comply JUMP.
# BeaverTails = harm in the TEXT -> expect flat (control).
python code/run_candidates.py --model llava_cot_tis --dataset figstep --n 10 \
    --sweep none:0,noise:10,noise:20,noise:30 --out results/fignoise/figstep
python code/run_candidates.py --model llava_cot_tis --dataset beavertails --n 10 \
    --sweep none:0,noise:10,noise:20,noise:30 --out results/fignoise/beaver

# ---- JUDGE: LLaMA-Guard ASR per condition (the locked judge), text-only ----
python code/judge_grid_guards.py --judge llamaguard \
    --input_dir results/fignoise/figstep --out_dir results/fignoise/figstep/guard
python code/judge_grid_guards.py --judge llamaguard \
    --input_dir results/fignoise/beaver --out_dir results/fignoise/beaver/guard