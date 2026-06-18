#!/bin/bash
# ============================================================================
# submit_msr_corruptions.sh — Task 1: extend the MSR-Align robustness sweep to
# TWO new corruption types beyond Gaussian blur/noise: JPEG compression and
# motion blur, each at 20% and 40% (clean is already covered by submit_msr_guard.sh).
#
# Model: base+MSR only (LLaVA-CoT Xkev/Llama-3.2V-11B-cot + MSR-Align LoRA).
#   ASR (FigStep)       -> Llama Guard 3 Vision   (judge_figstep_guard.py)
#   ORR (XSTest + MMSA) -> LLaMA-3-8B refusal judge (judge_safety_hf.py --mode orr)
#   SQA (ScienceQA)     -> LLaMA-3-8B utility judge  (judge_sqa_utility_hf.py)
# Exactly the judges/metrics already used for the blur sweep.
#
# 4 conditions: jpeg20 jpeg40 motion_blur20 motion_blur40
#   ASR/ORR outputs -> results/msr_guard_eval/<cond>/   (alongside clean/blur20/blur40)
#   SQA outputs      -> results/sqa_jpeg_pct/  and  results/sqa_motion_blur_pct/
#
# DAG (slurm afterok): each judge fires the moment its inference succeeds.
#   22 jobs total: 4 figstep-inf +4 asr, 4 orr-inf +4 orr-judge, 4 sqa-inf +2 sqa-judge.
#
# CLEAN is NOT re-run here — the clean MSR ASR/ORR/SQA already exist
# (results/msr_guard_eval/clean/ and results/sqa_noise_sweep/judged_base_msr_clean.json).
#
# Everything runs OFFLINE. Submit from the login node:
#     bash slurm_scripts/submit_msr_corruptions.sh
# When all done:  python code/report_msr_corruptions.py
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs results/msr_guard_eval results/sqa_jpeg_pct results/sqa_motion_blur_pct

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

# FREE preemptable partition (GPU billed 0.0 — does NOT drain the cohort pool).
COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

# submit <jobname> <timelimit> <dependency-jobid-or-empty> <python-command> -> echoes new job id
submit() {
  local name="$1" tlimit="$2" dep="$3" cmd="$4"
  local depflag=""
  if [ -n "$dep" ]; then depflag="--dependency=afterok:$dep"; fi
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" $depflag \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

echo "======================================================================"
echo "  submit_msr_corruptions.sh — MSR-Align: jpeg + motion_blur @ 20/40"
echo "======================================================================"

for CORR in jpeg motion_blur; do
  SQA_DIR="results/sqa_${CORR}_pct"
  SQA_DEPS=""
  for P in 20 40; do
    COND="${CORR}${P}"
    DIR="results/msr_guard_eval/${COND}"

    # --- ASR: FigStep inference -> Llama Guard 3 Vision judge ---
    FIG=$(submit "msrc_${COND}_figinf" 10:00:00 "" \
      "python code/eval_msr_guard.py --task figstep --corrupt ${CORR} --corrupt_pct ${P}")
    submit "msrc_${COND}_asr" 3:00:00 "$FIG" \
      "python code/judge_figstep_guard.py ${DIR}/responses_figstep.json --out_dir ${DIR}"

    # --- ORR: XSTest+MMSA inference -> LLaMA-3 refusal judge ---
    ORR=$(submit "msrc_${COND}_orrinf" 12:00:00 "" \
      "python code/eval_msr_guard.py --task orr --corrupt ${CORR} --corrupt_pct ${P}")
    submit "msrc_${COND}_orrj" 3:00:00 "$ORR" \
      "python code/judge_safety_hf.py --mode orr ${DIR}/responses_orr.csv \
       && python code/format_orr_results.py ${DIR}"

    # --- SQA: ScienceQA inference (judged together per corruption below) ---
    SQA=$(submit "msrc_${COND}_sqainf" 5:00:00 "" \
      "python code/eval_sqa_noise_sweep.py --use_msr --corrupt ${CORR} --corrupt_pct ${P}")
    SQA_DEPS="${SQA_DEPS:+${SQA_DEPS}:}${SQA}"

    echo "  ${COND}: figinf=${FIG}  orrinf=${ORR}  sqainf=${SQA}"
  done

  # One SQA judge per corruption, after BOTH pct inferences (idempotent --skip-existing).
  submit "msrc_${CORR}_sqaj" 3:00:00 "$SQA_DEPS" \
    "python code/judge_sqa_utility_hf.py --dir ${SQA_DIR} --skip-existing"
  echo "  ${CORR} sqa judge depends on: ${SQA_DEPS}"
done

echo ""
echo "======================================================================"
echo "22 jobs submitted (jpeg + motion_blur @ 20/40; ASR+ORR+SQA each)."
echo "Watch:   squeue -u \$USER | grep msrc"
echo "When all done:  python code/report_msr_corruptions.py"
echo "======================================================================"
