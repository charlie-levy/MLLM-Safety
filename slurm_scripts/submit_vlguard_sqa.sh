#!/bin/bash
# ============================================================================
# ScienceQA utility eval for VLGuard mixed + posthoc (clean only).
#
# Per variant: infer ScienceQA-250 -> LLaMA-3 judge -> judged_*.json
#
#   inf_sqa_<v>  -->  judge_sqa_<v>
#
# Accuracy is in results/vlguard_eval/<v>/clean/judged_vlguard_<v>_sqa.json
#
#   bash slurm_scripts/submit_vlguard_sqa.sh
# When done pull and read the accuracy:
#   python -c "
#   import json, glob
#   for p in sorted(glob.glob('results_newton/vlguard_eval/*/clean/judged_vlguard_*_sqa.json')):
#       d = json.load(open(p)); print(p.split('/')[-3], d['accuracy'], d['correct'], d['total'])
#   "
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

COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

submit() {
  local name="$1" tlimit="$2" dep="$3" cmd="$4"
  local depflag=""
  if [ -n "$dep" ]; then depflag="--dependency=afterok:$dep"; fi
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" $depflag \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

echo "──────────────────────────────────────────────────────────────"
for V in mixed posthoc; do
  DIR="results/vlguard_eval/${V}/clean"
  JSONL="${DIR}/raw_vlguard_${V}_sqa.jsonl"

  INF=$(submit "vg_${V}_sqa_inf" 4:00:00 "" \
       "python code/eval_vlguard.py --variant ${V} --task sqa --blur_pct 0")

  JDG=$(submit "vg_${V}_sqa_jdg" 3:00:00 "$INF" \
       "python code/judge_sqa_utility_hf.py ${JSONL}")

  echo "${V}: inf=$INF  judge=$JDG"
done
echo "──────────────────────────────────────────────────────────────"
echo "Watch:  squeue -u \$USER | grep vg_"
echo "Pull when done, then:"
echo "  python -c \""
echo "  import json, glob"
echo "  for p in sorted(glob.glob('results_newton/vlguard_eval/*/clean/judged_vlguard_*_sqa.json')):"
echo "      d = json.load(open(p)); print(p.split('/')[-3], d['accuracy'], d['correct'], d['total'])"
echo "  \""
echo "──────────────────────────────────────────────────────────────"
