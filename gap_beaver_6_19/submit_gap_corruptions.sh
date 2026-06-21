#!/bin/bash
# ============================================================================
# submit_gap_corruptions.sh — BeaverTails-V ASR under 3 NEW corruption families.
#
#   motion_blur (directional, vs the isotropic gaussian blur)
#   jpeg        (compression artifacts — text SURVIVES)
#   pixelate    (downscale->upscale low-res — text destroyed at high levels)
#
# Each at 20% and 40%, for BOTH models (base, +TIS) = 12 cells. Full n=1180, the
# SAME prepared set + the SAME proven generate/scoring path that produced every
# existing cell (run_beavertails.py; only the COND map was extended). Each cell ->
# ONE JSON list of 1180 entries with full model_response + is_refusal + attack_success.
#
#   bash gap_beaver_6_19/submit_gap_corruptions.sh pilot   # 1 small job: eyeball the 3 corruptions
#   bash gap_beaver_6_19/submit_gap_corruptions.sh         # the 12 full jobs
#
# BILLED H100 (normal partition) for SPEED, all cells in PARALLEL. ~3-3.5 h/cell, no
# preemption (runs straight to completion). Complete cells skip instantly via the
# 1180-guard; incremental save + resume recover any node failure (re-run to heal).
# evc42 (known-bad) excluded.
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs gap_beaver_6_19/outputs
MODE="${1:-full}"

# login `python` is Python 2 (chokes on UTF-8) -> use the REU env for the preflight too
set +u
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
set -u

echo "preflight (CPU, free) ..."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python gap_beaver_6_19/preflight_gap_corruptions.py \
  || { echo ">>> PREFLIGHT FAILED — NOT submitting."; exit 1; }

ENVBLOCK='source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval'

# Est. wall-clock/job: ~9-11 s/img x 1180 ≈ ~3-3.5 h (H100, batch 1). All 12 run in
# parallel, so wall-clock ≈ one job (subject to normal-queue availability).
# BILLED H100 (normal partition): fast, no preemption. Proven directive — matches the
# repo's billed jobs (slurm_scripts): partition=normal + nvidia_h100_pcie + mem=80G +
# exclude=evc42. No --qos / --account on normal (cohort account is the default).
COMMON="--partition=normal --gres=gpu:nvidia_h100_pcie:1 --mem=80G --cpus-per-task=4 --exclude=evc42"
RUN="python gap_beaver_6_19/run_beavertails.py"
OUT="gap_beaver_6_19/outputs"

# ---- PILOT: one small job, 4 conditions x 5 images, saves the corrupted sample png ----
# strongest level (40%) of each new type so the corruption is obvious, + one TIS sanity.
if [ "$MODE" = "pilot" ]; then
  JID=$(sbatch --parsable $COMMON --job-name=gap_corr_pilot --time=01:30:00 \
    --output="logs/gap_corr_pilot_%j.log" --wrap="${ENVBLOCK}
${RUN} --model base --condition motion_blur40 --pilot || exit 1
${RUN} --model base --condition jpeg40        --pilot || exit 1
${RUN} --model base --condition pixelate40     --pilot || exit 1
${RUN} --model tis  --condition motion_blur40 --pilot || exit 1")
  echo "submitted PILOT job $JID  (BILLED H100, normal partition)"
  echo "watch:   squeue -u \$USER | grep gap_corr_pilot   |   tail -f logs/gap_corr_pilot_${JID}.log"
  echo "eyeball: gap_beaver_6_19/pilot_{motion_blur40,jpeg40,pixelate40}_sample.png"
  echo ">>> review the pilot, THEN run: bash gap_beaver_6_19/submit_gap_corruptions.sh"
  exit 0
fi

# ---- FULL: 12 jobs in PARALLEL (no --dependency: each cell is independent; re-running
# heals any that died — complete cells (1180) skip instantly via the guard). On the
# normal partition jobs run to completion (no preemption). ----
# submit_one <jobname> <model> <condition>
submit_one() {
  local name="$1"; local model="$2"; local cond="$3"
  local outf="${OUT}/results_beavertails_${model}_${cond}.json"
  sbatch --parsable $COMMON --job-name="$name" --time=05:00:00 \
    --output="logs/${name}_%j.log" --wrap="${ENVBLOCK}
N=\$(python -c \"import json;print(len(json.load(open('${outf}'))))\" 2>/dev/null || echo 0)
if [ \"\$N\" = \"1180\" ]; then echo '[skip] ${outf} already complete (1180)'; exit 0; fi
echo \"[run] ${model}/${cond}: \$N/1180 done, (re)starting (run_beavertails resumes)\"
${RUN} --model ${model} --condition ${cond} || exit 1"
}

#                jobname     model condition
J01=$(submit_one b_mb20  base motion_blur20)
J02=$(submit_one b_mb40  base motion_blur40)
J03=$(submit_one b_jp20  base jpeg20)
J04=$(submit_one b_jp40  base jpeg40)
J05=$(submit_one b_px20  base pixelate20)
J06=$(submit_one b_px40  base pixelate40)
J07=$(submit_one t_mb20  tis  motion_blur20)
J08=$(submit_one t_mb40  tis  motion_blur40)
J09=$(submit_one t_jp20  tis  jpeg20)
J10=$(submit_one t_jp40  tis  jpeg40)
J11=$(submit_one t_px20  tis  pixelate20)
J12=$(submit_one t_px40  tis  pixelate40)

echo "submitted 12 jobs (PARALLEL, BILLED H100 normal; complete cells skip instantly):"
echo "  base : $J01 mb20  $J02 mb40  $J03 jp20  $J04 jp40  $J05 px20  $J06 px40"
echo "  tis  : $J07 mb20  $J08 mb40  $J09 jp20  $J10 jp40  $J11 px20  $J12 px40"
echo
echo "watch:   squeue -u \$USER"
echo "grid:    python gap_beaver_6_19/calc_asr_corruptions.py   (any time)"
echo "heal:    re-run this script — complete cells (1180) skip, preempted ones resume"
