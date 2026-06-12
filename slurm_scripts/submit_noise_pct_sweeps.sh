#!/bin/bash
# Percentage-noise sweeps (0-100% scale, see code/noise_utils.py) for the
# per-model noise-% plots. Runs TIS, SAGE, MSR x ASR/ORR/SQA at noise
# 20/40/60/80/100%. 0% (clean) is reused from the existing clean baselines.
#   3 models x 3 metrics x 5 levels = 45 jobs.
# ORR/FigStep get 8h; SQA 4h.
# Usage: bash slurm_scripts/submit_noise_pct_sweeps.sh

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
cd /home/ch169788/llava_cot_eval
python code/$4
"
  echo "Submitted: $1"
}

for SPEC in "tis:--use_tis" "sage:--use_sage" "msr:--use_msr"; do
  SHORT="${SPEC%%:*}"
  FLAG="${SPEC##*:}"
  for P in 20 40 60 80 100; do
    submit "${SHORT}_npf_${P}" "${SHORT}_figstep_noisepct_${P}" 8 \
           "eval_figstep_noise_sweep.py ${FLAG} --noise_pct ${P}"
    submit "${SHORT}_npo_${P}" "${SHORT}_orr_noisepct_${P}" 8 \
           "eval_orr_noise_sweep.py ${FLAG} --noise_pct ${P}"
    submit "${SHORT}_nps_${P}" "${SHORT}_sqa_noisepct_${P}" 4 \
           "eval_sqa_noise_sweep.py ${FLAG} --noise_pct ${P}"
  done
done

echo ""
echo "Submitted 45 noise-% sweep jobs. After they finish:"
echo "  - judge SQA:  bash slurm_scripts/submit_judge_noise_pct.sh"
echo "  - then pull + regenerate plots on your Mac."
