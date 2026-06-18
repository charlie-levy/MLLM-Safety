#!/bin/bash
# ============================================================================
# MSR-Align safety eval — FRESH inference (clean + 20% blur), model-based judges
# ============================================================================
# Model: base+MSR only (LLaVA-CoT Xkev/Llama-3.2V-11B-cot + MSR-Align LoRA).
# ASR (FigStep)        -> Llama Guard 3 Vision  (meta-llama/Llama-Guard-3-11B-Vision)
# ORR (XSTest + MMSA)  -> LLaMA-3-8B refusal judge (judge_safety_hf.py --mode orr)
#
# Auto-orchestrated DAG (slurm afterok dependencies):
#     inf_clean_figstep  ──> asr_guard_clean
#     inf_clean_orr      ──> orr_judge_clean
#     inf_blur20_figstep ──> asr_guard_blur20
#     inf_blur20_orr     ──> orr_judge_blur20
# The 4 inference jobs run in PARALLEL on separate H100s; each judge fires the
# moment its inference succeeds. When all 8 finish:
#     python code/report_msr_guard.py
#
# Everything runs OFFLINE (weights pre-cached). Submit from the login node:
#     bash slurm_scripts/submit_msr_guard.sh
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs results/msr_guard_eval

# Single-quoted: $PYTHONPATH stays literal and expands at runtime on the node.
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

COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

# submit <jobname> <timelimit> <dependency-jobid-or-empty> <python-command>
# Echoes the new job id (via --parsable).
submit() {
  local name="$1" tlimit="$2" dep="$3" cmd="$4"
  local depflag=""
  if [ -n "$dep" ]; then depflag="--dependency=afterok:$dep"; fi
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" $depflag \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

# ---- inference (parallel, no dependencies) ----
CF=$(submit msrg_inf_clean_figstep  10:00:00 "" "python code/eval_msr_guard.py --task figstep --blur_pct 0")
CO=$(submit msrg_inf_clean_orr      12:00:00 "" "python code/eval_msr_guard.py --task orr     --blur_pct 0")
BF=$(submit msrg_inf_blur20_figstep 10:00:00 "" "python code/eval_msr_guard.py --task figstep --blur_pct 20")
BO=$(submit msrg_inf_blur20_orr     12:00:00 "" "python code/eval_msr_guard.py --task orr     --blur_pct 20")

# ---- judges (each depends on its own inference job) ----
AC=$(submit msrg_asr_clean  3:00:00 "$CF" "python code/judge_figstep_guard.py results/msr_guard_eval/clean/responses_figstep.json  --out_dir results/msr_guard_eval/clean")
OC=$(submit msrg_orr_clean  3:00:00 "$CO" "python code/judge_safety_hf.py --mode orr results/msr_guard_eval/clean/responses_orr.csv && python code/format_orr_results.py results/msr_guard_eval/clean")
AB=$(submit msrg_asr_blur20 3:00:00 "$BF" "python code/judge_figstep_guard.py results/msr_guard_eval/blur20/responses_figstep.json --out_dir results/msr_guard_eval/blur20")
OB=$(submit msrg_orr_blur20 3:00:00 "$BO" "python code/judge_safety_hf.py --mode orr results/msr_guard_eval/blur20/responses_orr.csv && python code/format_orr_results.py results/msr_guard_eval/blur20")

echo "──────────────────────────────────────────────────────────────"
echo "Inference:  clean(figstep=$CF orr=$CO)  blur20(figstep=$BF orr=$BO)"
echo "Judges:     asr_clean=$AC orr_clean=$OC  asr_blur20=$AB orr_blur20=$OB"
echo "Judges auto-run on afterok. Watch:  squeue -u \$USER | grep msrg"
echo "When all done:  python code/report_msr_guard.py"
echo "──────────────────────────────────────────────────────────────"
