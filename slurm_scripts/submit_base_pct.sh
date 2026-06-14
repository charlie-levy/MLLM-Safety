#!/bin/bash
# BASE model (no safety adapter) — the "dangerously jailbreakable" baseline that
# makes the adapter contrast land. FigStep ASR / ORR / SQA at noise 0-80% and
# blur 20-80% (0% clean reused for blur). 3 metrics x (5 noise + 4 blur) = 27 jobs.
# Scored with the FIXED detector; re-score from responses afterwards as a safety net.
# Usage: bash slurm_scripts/submit_base_pct.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=jobname  $2=logname  $3=hours  $4=python args
  sbatch --job-name="$1" --partition=normal --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G --time="$3:00:00" --exclude=evc42 --output="logs/$2.log" --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ch169788/llava_cot_eval
python code/$4
"
  echo "Submitted: $1"
}

# NOISE (include 0% clean), then BLUR (20-80%). No adapter flag => Base.
for P in 0 20 40 60 80; do
  submit "base_npf_${P}" "base_figstep_noisepct_${P}" 8 "eval_figstep_noise_sweep.py --noise_pct ${P}"
  submit "base_npo_${P}" "base_orr_noisepct_${P}"     8 "eval_orr_noise_sweep.py --noise_pct ${P}"
  submit "base_nps_${P}" "base_sqa_noisepct_${P}"      4 "eval_sqa_noise_sweep.py --noise_pct ${P}"
done
for P in 20 40 60 80; do
  submit "base_bpf_${P}" "base_figstep_blurpct_${P}" 8 "eval_figstep_noise_sweep.py --blur_pct ${P}"
  submit "base_bpo_${P}" "base_orr_blurpct_${P}"     8 "eval_orr_noise_sweep.py --blur_pct ${P}"
  submit "base_bps_${P}" "base_sqa_blurpct_${P}"      4 "eval_sqa_noise_sweep.py --blur_pct ${P}"
done

echo ""
echo "Submitted 27 BASE jobs. After they finish:"
echo "  bash slurm_scripts/submit_judge_noise_pct.sh   # if base SQA noise needs judging"
echo "  bash slurm_scripts/submit_judge_blur_pct.sh    # base SQA blur"
echo "  python3 code/rescore_from_responses.py         # then pull + regenerate"
