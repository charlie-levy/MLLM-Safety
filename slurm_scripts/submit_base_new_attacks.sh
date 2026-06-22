#!/bin/bash
# ============================================================================
# submit_base_new_attacks.sh — run the NON-REASONING base model
# (meta-llama/Llama-3.2-11B-Vision-Instruct) on BeaverTails-V (1180) and SIUO
# (167) at BLUR 20%, saving FULL responses.
#
# This is the base-Llama counterpart to the LLaVA-CoT attack grid: the existing
# eval_attack_dataset.py / run_beavertails.py "base" is actually LLaVA-CoT (the
# REASONING model). Here we run the genuinely non-reasoning base model via the
# SAME proven load/generate path as eval_base_vision.py (reused, not changed).
#
# 2 jobs in PARALLEL (beavertails, siuo). Full sets, blur20 only. Incremental
# save every 50 + resume, so a node failure loses <=50 and re-running heals it.
#
#   bash slurm_scripts/submit_base_new_attacks.sh pilot   # smoke test: 5 imgs each
#   bash slurm_scripts/submit_base_new_attacks.sh         # the 2 full jobs
#
# BILLED H100 (normal partition) for SPEED — no preemption, runs straight to
# completion. Base (no CoT, 512 tokens) is fast: BeaverTails ~1-1.5 h, SIUO ~10 min.
# PREREQUISITE: datasets already materialized (they are — the LLaVA-CoT grid used
# them): datasets/new_attacks/{beavertails,siuo}/*.json
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs results/base_vision_new_attacks
MODE="${1:-full}"

# Guard: refuse to submit if the datasets were not materialized.
for DS in beavertails siuo; do
  if [ ! -f "datasets/new_attacks/${DS}/${DS}.json" ]; then
    echo "ERROR: datasets/new_attacks/${DS}/${DS}.json missing."
    echo "Run first (login node, online):  python code/prepare_new_attack_datasets.py"
    exit 1
  fi
done

# Cheap CPU preflight (login-safe): the driver must at least parse.
set +u
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
set -u
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -c "import ast; ast.parse(open('code/eval_base_new_attacks.py').read()); print('preflight OK: driver parses')" \
  || { echo ">>> PREFLIGHT FAILED — NOT submitting."; exit 1; }

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

# BILLED H100 (normal partition): fast, no preemption, runs straight to completion.
# Proven directive — matches the repo's billed jobs (partition=normal +
# nvidia_h100_pcie + mem=80G + exclude=evc42; no --qos/--account on normal).
COMMON="--partition=normal --gres=gpu:nvidia_h100_pcie:1 --mem=80G --cpus-per-task=4 --exclude=evc42"
RUN="python code/eval_base_new_attacks.py --blur_pct 20"

if [ "$MODE" = "pilot" ]; then
  JID=$(sbatch --parsable $COMMON --job-name=base_atk_pilot --time=00:40:00 \
    --output="logs/base_atk_pilot_%j.log" --wrap="${ENVBLOCK}
${RUN} --dataset beavertails --pilot || exit 1
${RUN} --dataset siuo        --pilot || exit 1")
  echo "submitted PILOT job $JID (BILLED H100, normal partition)"
  echo "watch: squeue -u \$USER | grep base_atk_pilot   |   tail -f logs/base_atk_pilot_${JID}.log"
  echo ">>> review, THEN run: bash slurm_scripts/submit_base_new_attacks.sh"
  exit 0
fi

submit_one() {
  local name="$1" ds="$2" tlimit="$3"
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" \
    --output="logs/${name}_%j.log" --wrap="${ENVBLOCK}
${RUN} --dataset ${ds} || exit 1"
}

J1=$(submit_one base_bt_blur20   beavertails 4:00:00)
J2=$(submit_one base_siuo_blur20 siuo        2:00:00)

echo "submitted 2 jobs (PARALLEL, BILLED H100 normal):"
echo "  beavertails : $J1   (blur20, 1180)"
echo "  siuo        : $J2   (blur20, 167)"
echo
echo "watch:  squeue -u \$USER"
echo "out:    results/base_vision_new_attacks/blur20/responses_{beavertails,siuo}.json"
echo "heal:   re-run this script — complete files skip, partial ones resume"
