#!/bin/bash
# ============================================================================
# Re-run ONLY XSTest for the VLGuard eval, after the load_xstest fix (deliver
# the safe request via a clean yes/no safety prompt instead of the stale
# "extract the text" OCR-bail prompt that collapsed ORR to ~0). MMSA + FigStep
# are unaffected and NOT re-run.
#
# Per variant/condition: re-infer XSTest (250) -> rebuild responses_orr.csv from
# the existing responses_mmsa.json -> re-judge ORR (LLaMA) -> reformat.
#
#   inf_xstest_<v>_<cond>  --> orr_judge_<v>_<cond>   (judge + format_orr_results)
#
#   bash slurm_scripts/submit_vlguard_xstest.sh
# When done:  OPENBLAS_NUM_THREADS=1 python code/report_vlguard.py
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

VARIANTS=(mixed posthoc)
PCTS=(0 20 40)

echo "──────────────────────────────────────────────────────────────"
for V in "${VARIANTS[@]}"; do
  for P in "${PCTS[@]}"; do
    if [ "$P" -eq 0 ]; then COND="clean"; else COND="blur${P}"; fi
    DIR="results/vlguard_eval/${V}/${COND}"
    if [ ! -f "${DIR}/responses_mmsa.json" ]; then
      echo "SKIP ${V}/${COND}: missing ${DIR}/responses_mmsa.json (run --task orr first)"
      continue
    fi
    XI=$(submit "vg_${V}_${COND}_xstest" 4:00:00 "" \
         "python code/eval_vlguard.py --variant ${V} --task xstest --blur_pct ${P}")
    OJ=$(submit "vg_${V}_${COND}_orrj" 3:00:00 "$XI" \
         "python code/judge_safety_hf.py --mode orr ${DIR}/responses_orr.csv && python code/format_orr_results.py ${DIR}")
    echo "${V}/${COND}: xstest=$XI  orr_judge=$OJ"
  done
done
echo "──────────────────────────────────────────────────────────────"
echo "Watch:  squeue -u \$USER | grep vg_"
echo "When done:  OPENBLAS_NUM_THREADS=1 python code/report_vlguard.py"
echo "──────────────────────────────────────────────────────────────"
