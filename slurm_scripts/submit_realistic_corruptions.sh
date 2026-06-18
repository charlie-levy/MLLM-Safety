#!/bin/bash
# Realistic-corruption sweeps: JPEG compression, low-light/brightness, and
# pixelation (see code/jpeg_utils.py, brightness_utils.py, pixelate_utils.py).
# For each: 3 adapters (TIS/SAGE/MSR) x 3 evals (FigStep ASR / ORR / SQA) x
# 4 levels (20/40/60/80%).  3 corruptions x 36 = 108 jobs.
# 0% (clean) is reused from the existing clean baselines, as with noise/blur.
# ORR/FigStep get 8h; SQA 4h. Fastest GPUs (H100 PCIe). HF offline (cached model).
# Usage: bash slurm_scripts/submit_realistic_corruptions.sh
#
# After they finish:
#   - judge SQA:  bash slurm_scripts/submit_judge_realistic.sh
#   - make example strips (see that script's header)
#   - then pull + regenerate analysis on your Mac.

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=jobname  $2=logname  $3=hours  $4=python args
  sbatch --job-name="$1" \
         --partition=preemptable --qos=preemptable \
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ch169788/llava_cot_eval
python code/$4
"
  echo "Submitted: $1"
}

for CORR in jpeg brightness pixelate; do
  for SPEC in "tis:--use_tis" "sage:--use_sage" "msr:--use_msr"; do
    SHORT="${SPEC%%:*}"; FLAG="${SPEC##*:}"
    for P in 20 40 60 80; do
      submit "${SHORT}_${CORR}f_${P}" "${SHORT}_figstep_${CORR}_${P}" 8 \
             "eval_figstep_noise_sweep.py ${FLAG} --corrupt ${CORR} --corrupt_pct ${P}"
      submit "${SHORT}_${CORR}o_${P}" "${SHORT}_orr_${CORR}_${P}" 8 \
             "eval_orr_noise_sweep.py ${FLAG} --corrupt ${CORR} --corrupt_pct ${P}"
      submit "${SHORT}_${CORR}s_${P}" "${SHORT}_sqa_${CORR}_${P}" 4 \
             "eval_sqa_noise_sweep.py ${FLAG} --corrupt ${CORR} --corrupt_pct ${P}"
    done
  done
done

echo ""
echo "Submitted 108 realistic-corruption jobs (jpeg/brightness/pixelate)."
echo "After they finish: bash slurm_scripts/submit_judge_realistic.sh"
