#!/bin/bash
# Judge the SQA raw responses from the percentage-noise sweep (results/sqa_noise_pct/).
# Usage: bash slurm_scripts/submit_judge_noise_pct.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

sbatch --job-name="judge_npct" \
       --partition=preemptable --qos=preemptable \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=64G \
       --time=4:00:00 \
       --exclude=evc42 \
       --output="logs/judge_noise_pct.log" \
       --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /home/ch169788/llava_cot_eval
python code/judge_sqa_utility_hf.py --dir results/sqa_noise_pct --skip-existing
"
echo "Submitted: judge_npct"
