#!/bin/bash
# ============================================================================
# 3 fully independent MSR-Align eval runs, ALL in parallel.
#
# Per run × condition:
#   figstep inference  -->  ASR judge (Llama Guard 3 Vision)
#   orr inference      -->  ORR judge (LLaMA-3-8B)  -->  format
#
# Outputs:  results/msr_guard_eval_run{1,2,3}/{clean,blur20}/
# When done:  python code/report_msr_multirun.py
#
# SQA pulled from existing single-run dirs (report_msr_multirun.py reads them).
#
#   bash slurm_scripts/submit_msr_multirun.sh
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
  local name="$1" tlimit="$2" dep="$3" cmd="$4"
  local depflag=""
  if [ -n "$dep" ]; then depflag="--dependency=afterok:$dep"; fi
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" $depflag \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

echo "Submitting 3 independent MSR-Align eval runs..."
echo ""

for R in 1 2 3; do
  for BP in 0 20; do
    COND="clean"
    if [ "$BP" -gt 0 ]; then COND="blur${BP}"; fi
    TAG="r${R}_${COND}"
    OUT="results/msr_guard_eval_run${R}/${COND}"

    # ---- Inference (parallel per run×condition) ----
    FIG=$(submit "msrm_fig_${TAG}" 8:00:00 "" \
      "python code/eval_msr_guard.py --task figstep --blur_pct ${BP} --run_id ${R}")

    ORR=$(submit "msrm_orr_${TAG}" 8:00:00 "" \
      "python code/eval_msr_guard.py --task orr --blur_pct ${BP} --run_id ${R}")

    # ---- Judges (afterok their respective inference) ----
    ASR=$(submit "msrm_asr_${TAG}" 3:00:00 "$FIG" \
      "python code/judge_figstep_guard.py ${OUT}/responses_figstep.json --out_dir ${OUT}")

    ORRJ=$(submit "msrm_orrj_${TAG}" 3:00:00 "$ORR" \
      "python code/judge_safety_hf.py --mode orr ${OUT}/responses_orr.csv \
       && python code/format_orr_results.py ${OUT}")

    echo "  Run ${R} ${COND}:  fig_infer=${FIG}  orr_infer=${ORR}  asr_judge=${ASR}  orr_judge+fmt=${ORRJ}"
  done
done

echo ""
echo "══════════════════════════════════════════════════════════"
echo "All 24 jobs submitted (12 inference + 12 judge/format)."
echo "Watch:   squeue -u \$USER | grep msrm"
echo "Report:  python code/report_msr_multirun.py"
echo "══════════════════════════════════════════════════════════"
