#!/bin/bash
# Percentage-BLUR sweeps (0-100% scale, see code/blur_utils.py) for the per-model
# blur-% bar charts. Runs TIS, SAGE, MSR x ASR/ORR/SQA at blur 20/40/60/80%.
# 0% (clean) is reused from the existing clean baselines; we stop at 80%.
#   3 models x 3 metrics x 4 levels = 36 jobs.
# ORR/FigStep get 8h; SQA 4h. Uses the fastest GPUs (H100 PCIe).
# Usage: bash slurm_scripts/submit_blur_pct_sweeps.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=jobname  $2=logname  $3=hours  $4=python args
  sbatch --job-name="$1" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time="$3:00:00" \
         --exclude=evc42 \
         --output="logs/$2.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
# Base model is cached and adapters are local, so load offline only -- guards
# against the flaky HF Hub call (\"client has been closed\") that killed the MSR runs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ch169788/llava_cot_eval
python code/$4
"
  echo "Submitted: $1"
}

for SPEC in "tis:--use_tis" "sage:--use_sage" "msr:--use_msr"; do
  SHORT="${SPEC%%:*}"
  FLAG="${SPEC##*:}"
  for P in 20 40 60 80; do
    submit "${SHORT}_bpf_${P}" "${SHORT}_figstep_blurpct_${P}" 8 \
           "eval_figstep_noise_sweep.py ${FLAG} --blur_pct ${P}"
    submit "${SHORT}_bpo_${P}" "${SHORT}_orr_blurpct_${P}" 8 \
           "eval_orr_noise_sweep.py ${FLAG} --blur_pct ${P}"
    submit "${SHORT}_bps_${P}" "${SHORT}_sqa_blurpct_${P}" 4 \
           "eval_sqa_noise_sweep.py ${FLAG} --blur_pct ${P}"
  done
done

echo ""
echo "Submitted 36 blur-% sweep jobs. After they finish:"
echo "  - judge SQA:  bash slurm_scripts/submit_judge_blur_pct.sh"
echo "  - then pull + regenerate plots on your Mac."
