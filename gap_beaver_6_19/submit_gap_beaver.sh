#!/bin/bash
# ============================================================================
# submit_gap_beaver.sh — fill EVERY empty cell of the BeaverTails-V ASR grid.
#
#   Existing (string-match ASR, n=1180):  base/clean 76.95   base/blur20 77.80
#                                          base+TIS/clean 21.36
#   This fills the 7 dashes (one parameterized driver, run_beavertails.py):
#       base       : noise20  noise40  blur40
#       base+TIS   : noise20  noise40  blur20  blur40
#
# Same proven method that made the existing numbers (the scorer reproduces
# 76.95/77.80/21.36 to the decimal). Each cell -> ONE JSON list of 1180 entries,
# each with full model_response + is_refusal + attack_success.
#
#   bash gap_beaver_6_19/submit_gap_beaver.sh pilot   # 1 small job: 4 pilots (5 imgs each)
#   bash gap_beaver_6_19/submit_gap_beaver.sh         # the 7 full jobs (after pilot sign-off)
#
# H100 on billed `normal` partition (~3x faster, reliable), cells run in PARALLEL
# at a time, V100-32GB. Each skips itself if its JSON already exists (resume the
# grid). Small + sequential = a modest, bounded amount of GPU.
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs gap_beaver_6_19/outputs
MODE="${1:-full}"

# The login node's bare `python` is Python 2 (chokes on UTF-8 in config.py). Use
# the REU env's Python 3 for the login-node preflight too. (set +u: conda's
# activate script references unset vars.)
set +u
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
set -u

# ---- PREFLIGHT on the login node (CPU, 0 GPU-hours). Gate the submit. ----
echo "preflight (CPU, free) ..."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python gap_beaver_6_19/preflight_gap_beaver.py \
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

# H100 on the billed `normal` partition (user choice): ~3x faster than V100, reliable
# (no preemption), IDENTICAL batch-1/bf16 decoding so the numbers are unchanged. The 4
# TIS cells run in parallel -> ~25-30 billed GPU-h total. Base cells already complete
# -> the guard skips them instantly regardless of partition.
# Est. wall-clock/job: ~9-10 s/img x 1180 ≈ ~3 h (H100, batch 1).
COMMON="--partition=normal --gres=gpu:nvidia_h100_pcie:1 --mem=40G --cpus-per-task=4 --exclude=evc42"
RUN="python gap_beaver_6_19/run_beavertails.py"
OUT="gap_beaver_6_19/outputs"

# ---- PILOT: one small job, 4 conditions x 5 images, saves sample pngs ----
if [ "$MODE" = "pilot" ]; then
  JID=$(sbatch --parsable $COMMON --job-name=gap_pilot --time=01:30:00 \
    --output="logs/gap_pilot_%j.log" --wrap="${ENVBLOCK}
${RUN} --model base --condition clean   --pilot || exit 1
${RUN} --model base --condition noise20 --pilot || exit 1
${RUN} --model base --condition blur40  --pilot || exit 1
${RUN} --model tis  --condition clean   --pilot || exit 1")
  echo "submitted PILOT job $JID  (FREE preemptable V100-32GB)"
  echo "watch:   squeue -u \$USER | grep gap_pilot   |   tail -f logs/gap_pilot_${JID}.log"
  echo "inspect: ${OUT}/results_beavertails_*_pilot.json   +   gap_beaver_6_19/pilot_*_sample.png"
  echo ">>> READ the pilots, confirm the ASR, THEN run: bash gap_beaver_6_19/submit_gap_beaver.sh"
  exit 0
fi

# ---- FULL: 7 jobs in PARALLEL (free partition -> concurrency is free; no orphan chain) ----
# submit_one <name> <model> <condition> [dep_jid]
submit_one() {
  local name="$1"; local model="$2"; local cond="$3"; local dep="${4:-}"
  local outf="${OUT}/results_beavertails_${model}_${cond}.json"
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterany:$dep"
  sbatch --parsable $COMMON $depflag --job-name="$name" --time=10:00:00 \
    --output="logs/${name}_%j.log" --wrap="${ENVBLOCK}
N=\$(python -c \"import json;print(len(json.load(open('${outf}'))))\" 2>/dev/null || echo 0)
if [ \"\$N\" = \"1180\" ]; then echo '[skip] ${outf} already complete (1180)'; exit 0; fi
echo \"[run] ${cond}: \$N/1180 done, (re)starting (run_beavertails resumes)\"
${RUN} --model ${model} --condition ${cond} || exit 1"
}

# PARALLEL (no --dependency): on the FREE partition concurrency costs $0, finishes
# ~4x faster than chaining, and kills the afterany orphan bug — a preempted cell no
# longer skips the rest, and each cell resumes independently on the next pass.
# Complete cells (1180) skip instantly via the guard, so re-running is cheap.
J1=$(submit_one gap_base_noise20 base noise20)
J2=$(submit_one gap_base_noise40 base noise40)
J3=$(submit_one gap_base_blur40  base blur40)
J4=$(submit_one gap_tis_noise20  tis  noise20)
J5=$(submit_one gap_tis_noise40  tis  noise40)
J6=$(submit_one gap_tis_blur20   tis  blur20)
J7=$(submit_one gap_tis_blur40   tis  blur40)

echo "submitted 7 jobs (PARALLEL, H100 on normal partition; complete cells skip instantly):"
echo "  $J1 gap_base_noise20   $J2 gap_base_noise40   $J3 gap_base_blur40"
echo "  $J4 gap_tis_noise20    $J5 gap_tis_noise40    $J6 gap_tis_blur20   $J7 gap_tis_blur40"
echo
echo "watch:   squeue -u \$USER"
echo "grid:    python gap_beaver_6_19/calc_asr_beavertails.py   (any time)"
