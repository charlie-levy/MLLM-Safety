#!/bin/bash
# ============================================================================
# Re-run ONLY XSTest for the MSR-Align eval, after the load_xstest fix that
# delivers the safe request as TEXT (robust to image blur). MMSA + FigStep are
# unaffected and NOT re-run.
#
# Per condition: re-infer XSTest (250) -> rebuild responses_orr.csv from the
# existing responses_mmsa.json -> re-judge ORR (LLaMA) -> reformat.
#
#   inf_xstest_clean  --> orr_judge_clean   (judge + format_orr_results)
#   inf_xstest_blur20 --> orr_judge_blur20
#
#   bash slurm_scripts/submit_msr_guard_xstest.sh
# When done:  python code/report_msr_guard.py
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

# ---- XSTest re-inference (parallel) ----
XC=$(submit msrg_xstest_clean  5:00:00 "" "python code/eval_msr_guard.py --task xstest --blur_pct 0")
XB=$(submit msrg_xstest_blur20 5:00:00 "" "python code/eval_msr_guard.py --task xstest --blur_pct 20")

# ---- ORR judge + format (depend on their inference) ----
OC=$(submit msrg_orr_clean_re  3:00:00 "$XC" "python code/judge_safety_hf.py --mode orr results/msr_guard_eval/clean/responses_orr.csv  && python code/format_orr_results.py results/msr_guard_eval/clean")
OB=$(submit msrg_orr_blur20_re 3:00:00 "$XB" "python code/judge_safety_hf.py --mode orr results/msr_guard_eval/blur20/responses_orr.csv && python code/format_orr_results.py results/msr_guard_eval/blur20")

echo "──────────────────────────────────────────────────────────────"
echo "XSTest re-inference:  clean=$XC  blur20=$XB"
echo "ORR judge+format:     clean=$OC  blur20=$OB  (auto-run on afterok)"
echo "Watch:  squeue -u \$USER | grep msrg"
echo "When done:  python code/report_msr_guard.py"
echo "──────────────────────────────────────────────────────────────"
