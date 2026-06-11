#!/bin/bash
# MSR-Align full corruption sweeps (to match Base/TIS coverage):
#   noise + blur  x  ASR + ORR + SQA  x  sev1-5  = 30 jobs
# ORR jobs get 8h (XSTest+MMSA ~6h); FigStep 8h; SQA 4h.
# NOTE: MSR clean ASR/ORR are submitted separately (submit_msr.sh) — this is
#       only the severity 1-5 sweeps.
# Usage: bash slurm_scripts/submit_msr_sweeps.sh

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

for NOISE in gaussian_noise gaussian_blur; do
  SHORT=$([ "$NOISE" = "gaussian_noise" ] && echo "noise" || echo "blur")
  for SEV in 1 2 3 4 5; do
    submit "msr_fig_${SHORT:0:1}${SEV}" "msr_figstep_${SHORT}_sev${SEV}" 8 \
           "eval_figstep_noise_sweep.py --use_msr --severity ${SEV} --noise_type ${NOISE}"
    submit "msr_orr_${SHORT:0:1}${SEV}" "msr_orr_${SHORT}_sev${SEV}" 8 \
           "eval_orr_noise_sweep.py --use_msr --severity ${SEV} --noise_type ${NOISE}"
    submit "msr_sqa_${SHORT:0:1}${SEV}" "msr_sqa_${SHORT}_sev${SEV}" 4 \
           "eval_sqa_noise_sweep.py --use_msr --severity ${SEV} --noise_type ${NOISE}"
  done
done

echo ""
echo "Submitted 30 MSR sweep jobs. After they finish:"
echo "  - re-run the SQA judge on Newton:  bash slurm_scripts/submit_judge_sqa.sh"
echo "  - then pull + regenerate on your Mac."
