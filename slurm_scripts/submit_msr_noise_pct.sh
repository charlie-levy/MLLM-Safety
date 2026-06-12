#!/bin/bash
# Re-submit ONLY the MSR noise-% jobs (these produced no output in the original
# 45-job batch, while TIS and SAGE succeeded). MSR x ASR/ORR/SQA x 20/40/60/80%.
# We cap the study at 80%, so p100 is intentionally omitted -> 3 metrics x 4 levels
# = 12 jobs. ORR/FigStep get 8h; SQA 4h.
# Usage: bash slurm_scripts/submit_msr_noise_pct.sh
#
# After they finish:
#   - judge SQA:  bash slurm_scripts/submit_judge_noise_pct.sh
#   - then pull + regenerate plots on your Mac.

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
# The original MSR runs died on a flaky HF Hub call (\"client has been closed\").
# The base model is already cached (TIS/SAGE ran fine) and the MSR adapter is a
# local path, so force offline: load from cache only, never touch the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ch169788/llava_cot_eval
python code/$4
"
  echo "Submitted: $1"
}

for P in 20 40 60 80; do
  submit "msr_npf_${P}" "msr_figstep_noisepct_${P}" 8 \
         "eval_figstep_noise_sweep.py --use_msr --noise_pct ${P}"
  submit "msr_npo_${P}" "msr_orr_noisepct_${P}" 8 \
         "eval_orr_noise_sweep.py --use_msr --noise_pct ${P}"
  submit "msr_nps_${P}" "msr_sqa_noisepct_${P}" 4 \
         "eval_sqa_noise_sweep.py --use_msr --noise_pct ${P}"
done

echo ""
echo "Submitted 12 MSR noise-% jobs (p20/40/60/80 x figstep/orr/sqa)."
echo "After they finish:"
echo "  - judge SQA:  bash slurm_scripts/submit_judge_noise_pct.sh"
echo "  - then pull + regenerate plots on your Mac."
