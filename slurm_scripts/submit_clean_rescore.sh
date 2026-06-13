#!/bin/bash
# Regenerate the CLEAN (0%) FigStep + ORR baselines for all 3 adapters with the
# FIXED refusal detector (curly-apostrophe bug). --noise_pct 0 leaves the image
# untouched, so this is the clean baseline; it writes p0 metric JSONs AND the
# per-sample response CSVs (so they are re-scorable like every other level):
#   results/figstep_noise_pct/asr_<m>_gaussian_noise_pct_p0.json
#   results/orr_noise_pct/orr_<m>_gaussian_noise_pct_p0.json
# SQA clean is NOT affected by the bug, so it is not re-run here.
# 6 jobs (figstep + orr x tis/sage/msr). Usage: bash slurm_scripts/submit_clean_rescore.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=jobname  $2=logname  $3=python args
  sbatch --job-name="$1" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --exclude=evc42 \
         --output="logs/$2.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ch169788/llava_cot_eval
python code/$3
"
  echo "Submitted: $1"
}

for SPEC in "tis:--use_tis" "sage:--use_sage" "msr:--use_msr"; do
  SHORT="${SPEC%%:*}"; FLAG="${SPEC##*:}"
  submit "${SHORT}_figclean" "${SHORT}_figstep_clean_p0" "eval_figstep_noise_sweep.py ${FLAG} --noise_pct 0"
  submit "${SHORT}_orrclean" "${SHORT}_orr_clean_p0"     "eval_orr_noise_sweep.py ${FLAG} --noise_pct 0"
done

echo ""
echo "Submitted 6 clean-baseline jobs (figstep + orr x 3 models, fixed detector)."
