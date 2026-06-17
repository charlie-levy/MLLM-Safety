#!/bin/bash
# ============================================================================
# Baseline eval: meta-llama/Llama-3.2-11B-Vision-Instruct (no safety, no CoT)
# FigStep ASR + XSTest/MMSA ORR + ScienceQA utility, blur_pct 0/20/40/60/80/100
# (= severity levels 0-5). All metrics are string-matching, no judge jobs.
# One job per blur level (model loaded once, all 4 datasets run in sequence).
#
#   bash slurm_scripts/submit_base_vision.sh
# When done pull results and run:
#   python code/report_base_vision.py
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

# Check model is cached before submitting
if [ ! -d "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct" ]; then
  echo "ERROR: meta-llama/Llama-3.2-11B-Vision-Instruct not found in HF cache."
  echo "Download it first on the login node:"
  echo "  HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 hf download meta-llama/Llama-3.2-11B-Vision-Instruct --max-workers 1"
  exit 1
fi

ENVBLOCK='source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval'

COMMON="--partition=normal --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

submit() {
  local name="$1" tlimit="$2" cmd="$3"
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

echo "──────────────────────────────────────────────────────────────"
echo "  Submitting base Vision-Instruct eval (blur_pct 0/20/40/60/80/100)"
echo "──────────────────────────────────────────────────────────────"

# Each job runs all 4 datasets: FigStep(500)+XSTest(250)+MMSA(428)+SQA(250) = 1428 samples
# Estimated ~3-4h per job on H100 at ~6 samples/min

for PCT in 0 20 40 60 80 100; do
  if [ "$PCT" -eq 0 ]; then COND="clean"; else COND="blur${PCT}"; fi
  JID=$(submit "bv_${COND}" 6:00:00 \
       "python code/eval_base_vision.py --blur_pct ${PCT}")
  echo "  blur_pct=${PCT} (${COND}): job=${JID}"
done

echo "──────────────────────────────────────────────────────────────"
echo "Watch:  squeue -u \$USER | grep bv_"
echo "When done, pull and run:  python code/report_base_vision.py"
echo "──────────────────────────────────────────────────────────────"
