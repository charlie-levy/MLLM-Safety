#!/bin/bash
# ============================================================================
# VLGuard (LLaVA-1.5-7B) safety eval — clean + 20% + 40% blur, model judges
# ============================================================================
# Variants: mixed, posthoc  (converted to HF format first — see PREREQS below).
# ASR (FigStep)        -> Llama Guard 3 Vision  (meta-llama/Llama-Guard-3-11B-Vision)
# ORR (XSTest + MMSA)  -> LLaMA-3-8B refusal judge (judge_safety_hf.py --mode orr)
#
# PREREQS (run once, see code/convert_vlguard_to_hf.py header):
#   login node:  hf download ys-zong/llava-v1.5-7b-Mixed-lora   --max-workers 1
#                hf download ys-zong/llava-v1.5-7b-Posthoc-lora --max-workers 1
#                hf download llava-hf/llava-1.5-7b-hf           --max-workers 1
#   srun GPU:    python code/convert_vlguard_to_hf.py --variant mixed
#                python code/convert_vlguard_to_hf.py --variant posthoc
#
# For each (variant, condition): 2 inference jobs run in PARALLEL on separate
# H100s; each judge fires on afterok of its inference. When all finish:
#     python code/report_vlguard.py
#
# Everything runs OFFLINE (weights pre-cached). Submit from the login node:
#     bash slurm_scripts/submit_vlguard.sh
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs results/vlguard_eval

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

    # ---- inference (parallel, no dependencies) ----
    IF=$(submit "vg_${V}_${COND}_fs" 4:00:00 "" \
         "python code/eval_vlguard.py --variant ${V} --task figstep --blur_pct ${P}")
    IO=$(submit "vg_${V}_${COND}_orr" 4:00:00 "" \
         "python code/eval_vlguard.py --variant ${V} --task orr --blur_pct ${P}")

    # ---- judges (afterok of their own inference) ----
    AJ=$(submit "vg_${V}_${COND}_asr" 3:00:00 "$IF" \
         "python code/judge_figstep_guard.py ${DIR}/responses_figstep.json --out_dir ${DIR}")
    OJ=$(submit "vg_${V}_${COND}_orrj" 3:00:00 "$IO" \
         "python code/judge_safety_hf.py --mode orr ${DIR}/responses_orr.csv && python code/format_orr_results.py ${DIR}")

    echo "${V}/${COND}: inf(fs=$IF orr=$IO) judges(asr=$AJ orr=$OJ)"
  done
done
echo "──────────────────────────────────────────────────────────────"
echo "24 jobs submitted (2 variants x 3 conditions x [2 inf + 2 judge])."
echo "Judges auto-run on afterok. Watch:  squeue -u \$USER | grep vg_"
echo "When all done:  python code/report_vlguard.py"
echo "──────────────────────────────────────────────────────────────"
