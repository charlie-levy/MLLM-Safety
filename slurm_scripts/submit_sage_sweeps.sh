#!/bin/bash
# Fill in SAGE's missing corruption sweeps:
#   - noise ORR  sev1-5   (noise ASR already done; noise SQA below)
#   - noise SQA  sev1-5
#   - blur  ASR  sev1-5
#   - blur  ORR  sev1-5
#   - blur  SQA  sev1-5
# 25 jobs. ORR jobs get 8h (XSTest+MMSA ~6h); FigStep 8h; SQA 4h.
# Usage: bash slurm_scripts/submit_sage_sweeps.sh

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

for SEV in 1 2 3 4 5; do
  # --- noise: ORR + SQA (ASR already done) ---
  submit "sage_orr_n${SEV}" "sage_orr_noise_sev${SEV}" 8 \
         "eval_orr_noise_sweep.py --use_sage --severity ${SEV} --noise_type gaussian_noise"
  submit "sage_sqa_n${SEV}" "sage_sqa_noise_sev${SEV}" 4 \
         "eval_sqa_noise_sweep.py --use_sage --severity ${SEV} --noise_type gaussian_noise"

  # --- blur: ASR + ORR + SQA ---
  submit "sage_fig_b${SEV}" "sage_figstep_blur_sev${SEV}" 8 \
         "eval_figstep_noise_sweep.py --use_sage --severity ${SEV} --noise_type gaussian_blur"
  submit "sage_orr_b${SEV}" "sage_orr_blur_sev${SEV}" 8 \
         "eval_orr_noise_sweep.py --use_sage --severity ${SEV} --noise_type gaussian_blur"
  submit "sage_sqa_b${SEV}" "sage_sqa_blur_sev${SEV}" 4 \
         "eval_sqa_noise_sweep.py --use_sage --severity ${SEV} --noise_type gaussian_blur"
done

echo ""
echo "Submitted 25 SAGE sweep jobs. After they finish:"
echo "  - re-run the SQA judge on Newton:  bash slurm_scripts/submit_judge_sqa.sh"
echo "  - then pull + regenerate on your Mac."
