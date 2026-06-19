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
# FREE preemptable V100-32GB (GPU billed 0.0). Full jobs are chained one-at-a-time,
# --requeue, and each skips itself if its JSON already exists (resume the grid).
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs gap_beaver_6_19/outputs
MODE="${1:-full}"

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

# FREE partition + idle V100-32GB (uncontended -> starts now, not preempted).
COMMON="--partition=preemptable --qos=preemptable --gres=gpu:tesla_v100-pcie-32gb:1 --mem=40G --requeue --exclude=evc42"
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
  echo "submitted PILOT job $JID  (FREE V100-32GB)"
  echo "watch:   squeue -u \$USER | grep gap_pilot   |   tail -f logs/gap_pilot_${JID}.log"
  echo "inspect: ${OUT}/results_beavertails_*_pilot.json   +   gap_beaver_6_19/pilot_*_sample.png"
  echo ">>> READ the pilots, confirm the ASR, THEN run: bash gap_beaver_6_19/submit_gap_beaver.sh"
  exit 0
fi

# ---- FULL: 7 jobs, chained one-at-a-time ----
# submit_one <name> <model> <condition> [dep_jid]
submit_one() {
  local name="$1"; local model="$2"; local cond="$3"; local dep="${4:-}"
  local outf="${OUT}/results_beavertails_${model}_${cond}.json"
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterany:$dep"
  sbatch --parsable $COMMON $depflag --job-name="$name" --time=10:00:00 \
    --output="logs/${name}_%j.log" --wrap="${ENVBLOCK}
if [ -f ${outf} ]; then echo '[skip] ${outf} exists'; exit 0; fi
${RUN} --model ${model} --condition ${cond} || exit 1"
}

J1=$(submit_one gap_base_noise20 base noise20)
J2=$(submit_one gap_base_noise40 base noise40 "$J1")
J3=$(submit_one gap_base_blur40  base blur40  "$J2")
J4=$(submit_one gap_tis_noise20  tis  noise20 "$J3")
J5=$(submit_one gap_tis_noise40  tis  noise40 "$J4")
J6=$(submit_one gap_tis_blur20   tis  blur20  "$J5")
J7=$(submit_one gap_tis_blur40   tis  blur40  "$J6")

echo "submitted 7 jobs (run strictly one-at-a-time, FREE preemptable V100-32GB):"
echo "  $J1 gap_base_noise20   $J2 gap_base_noise40   $J3 gap_base_blur40"
echo "  $J4 gap_tis_noise20    $J5 gap_tis_noise40    $J6 gap_tis_blur20   $J7 gap_tis_blur40"
echo
echo "watch:   squeue -u \$USER"
echo "grid:    python gap_beaver_6_19/calc_asr_beavertails.py   (any time)"
