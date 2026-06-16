#!/bin/bash
# ============================================================================
# VLGuard ASR judges ONLY — re-fire Llama Guard 3 Vision on existing FigStep
# responses (use when inference already ran but the ASR judges failed, e.g. the
# AutoModelForVision2Seq import bug fixed in c47680a). NO re-inference.
#
# Reads results/vlguard_eval/<variant>/<cond>/responses_figstep.json (already on
# disk) and writes asr_guard.json + figstep_results.json per condition.
#
#   bash slurm_scripts/submit_vlguard_asr.sh
#   # then, when the 6 jobs finish:
#   OPENBLAS_NUM_THREADS=1 python code/report_vlguard.py
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

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
  local name="$1" cmd="$2"
  sbatch --parsable $COMMON --job-name="$name" --time=3:00:00 \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

VARIANTS=(mixed posthoc)
PCTS=(0 20 40)

echo "──────────────────────────────────────────────────────────────"
for V in "${VARIANTS[@]}"; do
  for P in "${PCTS[@]}"; do
    if [ "$P" -eq 0 ]; then COND="clean"; else COND="blur${P}"; fi
    DIR="results/vlguard_eval/${V}/${COND}"
    RESP="${DIR}/responses_figstep.json"
    if [ ! -f "$RESP" ]; then
      echo "SKIP ${V}/${COND}: missing ${RESP} (run inference first)"
      continue
    fi
    AJ=$(submit "vg_${V}_${COND}_asr" \
         "python code/judge_figstep_guard.py ${RESP} --out_dir ${DIR}")
    echo "${V}/${COND}: asr judge = $AJ"
  done
done
echo "──────────────────────────────────────────────────────────────"
echo "Watch:  squeue -u \$USER | grep _asr"
echo "When done:  OPENBLAS_NUM_THREADS=1 python code/report_vlguard.py"
echo "──────────────────────────────────────────────────────────────"
