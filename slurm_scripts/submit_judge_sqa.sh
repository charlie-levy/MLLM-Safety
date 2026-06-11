#!/bin/bash
# Run the LLaMA-3 SQA utility judge ON NEWTON (GPU, transformers) instead of
# locally via Ollama. Judges every raw_*.jsonl that doesn't yet have a
# judged_*.json, for both the noise and blur sweeps, in one job.
#
# Usage: bash slurm_scripts/submit_judge_sqa.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

sbatch --job-name="judge_sqa" \
       --partition=normal \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=64G \
       --time=4:00:00 \
       --exclude=evc42 \
       --output="logs/judge_sqa.log" \
       --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/judge_sqa_utility_hf.py --dir results/sqa_noise_sweep --skip-existing
python code/judge_sqa_utility_hf.py --dir results/sqa_blur_sweep  --skip-existing
"
echo "Submitted: judge_sqa"
