#!/bin/bash
# Generate the CLEAN FigStep TIS per-sample responses (none were saved on the
# original clean run). --noise_pct 0 leaves the image untouched, so this is the
# clean baseline; it writes results/figstep_noise_pct/responses_base_tis_gaussian_noise_pct_p0.csv
# which combine_figstep_tis_responses.py picks up as "clean (0%)".
# Usage: bash slurm_scripts/submit_figstep_tis_clean_responses.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

sbatch --job-name="tis_fig_clean" \
       --partition=preemptable --qos=preemptable \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=64G \
       --time=2:00:00 \
       --exclude=evc42 \
       --output="logs/tis_figstep_clean_responses.log" \
       --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /home/ch169788/llava_cot_eval
python code/eval_figstep_noise_sweep.py --use_tis --noise_pct 0
"
echo "Submitted: tis_fig_clean"
