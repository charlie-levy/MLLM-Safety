#!/bin/bash
# Re-run the full ScienceQA sweep SAVING raw responses (raw_*.jsonl) so utility
# can be scored by the LLaMA-3 judge (code/judge_sqa_utility.py) instead of regex.
#
# 22 jobs: {base, base+tis} x {clean, noise sev1-5, blur sev1-5}
# Usage: bash slurm_scripts/submit_sqa_utility_sweep.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=job  $2=log  $3=python args
  sbatch --job-name="$1" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --output="logs/$2.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
python code/eval_sqa_noise_sweep.py $3
"
  echo "Submitted: $1"
}

for TIS in "" "--use_tis"; do
  TAG=$([ "$TIS" = "--use_tis" ] && echo "tis" || echo "base")

  # clean (severity 0)
  submit "sqau_${TAG}_clean" "sqau_${TAG}_clean" "$TIS --severity 0 --noise_type gaussian_noise"

  # noise + blur sev 1-5
  for NOISE in gaussian_noise gaussian_blur; do
    SHORT=$([ "$NOISE" = "gaussian_noise" ] && echo "noise" || echo "blur")
    for SEV in 1 2 3 4 5; do
      submit "sqau_${TAG}_${SHORT}_s${SEV}" "sqau_${TAG}_${SHORT}_sev${SEV}" \
             "$TIS --severity $SEV --noise_type $NOISE"
    done
  done
done
