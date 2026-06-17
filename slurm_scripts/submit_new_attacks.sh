#!/bin/bash
# ============================================================================
# submit_new_attacks.sh — Task 2: replace FigStep with three image-based safety
# attack datasets and measure ASR for LLaVA-CoT with and without TIS.
#
#   SIUO          167   sinwang/SIUO
#   BeaverTails-V 1180  PKU-Alignment/BeaverTails-V (20 categories, eval split)
#   SPA-VL        265   sqrti/SPA-VL (test/harm)
#
# Models: base (LLaVA-CoT) and base+TIS.  Conditions: clean / blur20 / blur40.
# ASR scored the SAME string-match way as the FigStep 70.4% / 13.8% numbers
# (compute_asr in eval_attack_dataset.py) — no judge jobs needed.
#
# 18 inference jobs = 3 datasets x 2 models x 3 conditions.
# Per-sample responses are saved so any number can be re-derived later.
#
# PREREQUISITE (run ONCE on the login node, online):
#     python code/prepare_new_attack_datasets.py
#
# Then submit (offline) from the login node:
#     bash slurm_scripts/submit_new_attacks.sh
# When all done:  python code/report_new_attacks.py
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs results/new_attacks

# Guard: refuse to submit if the datasets were not materialized yet.
for DS in siuo beavertails spavl; do
  if [ ! -f "datasets/new_attacks/${DS}/${DS}.json" ]; then
    echo "ERROR: datasets/new_attacks/${DS}/${DS}.json missing."
    echo "Run first:  python code/prepare_new_attack_datasets.py"
    exit 1
  fi
done

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

# Wall-time budget scales with dataset size (BeaverTails 1180 is the big one).
declare -A TLIMIT=( [siuo]="4:00:00" [spavl]="6:00:00" [beavertails]="16:00:00" )

echo "======================================================================"
echo "  submit_new_attacks.sh — SIUO / BeaverTails-V / SPA-VL  (base + TIS)"
echo "======================================================================"

for DS in siuo spavl beavertails; do
  for P in 0 20 40; do
    COND=$([ "$P" -eq 0 ] && echo clean || echo "blur${P}")

    B=$(submit "atk_${DS}_base_${COND}" "${TLIMIT[$DS]}" \
      "python code/eval_attack_dataset.py --dataset ${DS} --blur_pct ${P}")
    T=$(submit "atk_${DS}_tis_${COND}" "${TLIMIT[$DS]}" \
      "python code/eval_attack_dataset.py --dataset ${DS} --use_tis --blur_pct ${P}")

    echo "  ${DS} ${COND}: base=${B}  tis=${T}"
  done
done

echo ""
echo "======================================================================"
echo "18 jobs submitted (3 datasets x base/TIS x clean/blur20/blur40)."
echo "Watch:   squeue -u \$USER | grep atk_"
echo "When all done:  python code/report_new_attacks.py"
echo "======================================================================"
