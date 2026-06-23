#!/bin/bash
# ============================================================================
# Apples-to-apples ASR on model_responses/ with TWO text-only judges:
#   * LLaMA-Guard 3 Vision (text-only)  meta-llama/Llama-Guard-3-11B-Vision
#   * WildGuard                         allenai/wildguard
#
# Both judges see the same (prompt, response) text -> directly comparable ASR.
#
# DAG (slurm afterok) — judges run IN PARALLEL for speed:
#     preflight (8/file, BOTH models) ─┬─> full_llamaguard ─┐
#                                      └─> full_wildguard  ─┴─> report
# The preflight (~5 min) loads both guards and judges 8 real samples per file,
# so a model-load / chat-template / parse bug fails cheap. Then each judge runs
# on its OWN H100 concurrently.
#
# Partition: normal (NON-preemptable H100, won't get killed mid-run). NOTE: the
# normal partition BILLS against the GPU pool (preemptable is the free one).
# Runs OFFLINE (weights pre-cached). Submit from the login node:
#     bash slurm_scripts/submit_grid_guards.sh
#
# Prereq: WildGuard fully cached ->  hf download allenai/wildguard
#         (Llama-Guard-3-11B-Vision is already cached.)
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs results/grid_guard_eval

die() { echo "ERROR: $*" >&2; exit 1; }

# --- Sanity: grid present ---
N_JSON=$(ls model_responses/*.json 2>/dev/null | wc -l)
[ "$N_JSON" -gt 0 ] || die "no model_responses/*.json in $REPO (did you 'git pull'?)"

# --- Sanity: WildGuard COMPLETELY downloaded (this is what bit us once) ---
WG=~/.cache/huggingface/hub/models--allenai--wildguard
WG_SNAP=$(ls -d "$WG"/snapshots/*/ 2>/dev/null | head -1)
[ -n "$WG_SNAP" ] || die "WildGuard not cached. Run on login node:  hf download allenai/wildguard"
# Orphan *.incomplete temp blobs are harmless (abandoned multi-worker / pytorch_model.bin
# attempts). Warn, but let the authoritative file checks below decide loadability.
if ls "$WG"/blobs/*.incomplete >/dev/null 2>&1; then
  echo "WARN: orphan *.incomplete blobs present (harmless). Tidy with: rm $WG/blobs/*.incomplete"
fi
[ -e "${WG_SNAP}model.safetensors.index.json" ] || die "WildGuard missing model.safetensors.index.json — download incomplete"
{ [ -e "${WG_SNAP}tokenizer.model" ] || [ -e "${WG_SNAP}tokenizer.json" ]; } || die "WildGuard missing tokenizer (need tokenizer.model or tokenizer.json) — download incomplete"
# all safetensors shards present? (model-0000N-of-0000M -> need M shards)
SHARD1=$(ls "${WG_SNAP}"model-00001-of-*.safetensors 2>/dev/null | head -1)
[ -n "$SHARD1" ] || die "WildGuard has no sharded safetensors — download incomplete"
EXPECT=$(echo "$SHARD1" | sed -E 's/.*of-0*([0-9]+)\.safetensors/\1/')
HAVE=$(ls "${WG_SNAP}"model-*-of-*.safetensors 2>/dev/null | wc -l)
[ "$HAVE" -eq "$EXPECT" ] || die "WildGuard has $HAVE/$EXPECT safetensors shards — download incomplete. Re-run hf download."
# integrity: each shard must resolve (follow symlink) to a real, non-tiny file
for s in "${WG_SNAP}"model-*-of-*.safetensors; do
  SZ=$(du -L -m "$s" 2>/dev/null | cut -f1)
  { [ -n "$SZ" ] && [ "$SZ" -ge 1 ]; } || die "WildGuard shard $(basename "$s") is empty/dangling — re-download that file"
done
# WildGuard's tokenizer is a SentencePiece model; transformers 5.9 must convert it,
# which needs the sentencepiece lib (tiktoken is the fallback path). Fail fast here
# on the login node rather than crashing the WildGuard job on a compute node.
python -c "import sentencepiece" 2>/dev/null || die "sentencepiece not installed (WildGuard tokenizer needs it). Run:  pip install sentencepiece tiktoken"

echo "OK: $N_JSON response files + WildGuard ($HAVE/$EXPECT shards verified, index, tokenizer) + sentencepiece. Submitting..."

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

# normal = non-preemptable H100 (won't be killed mid-run). Default QOS for the partition.
COMMON="--partition=normal --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

submit() {  # submit <jobname> <timelimit> <dep-or-empty (colon-joined ids ok)> <python-command>
  local name="$1" tlimit="$2" dep="$3" cmd="$4"
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" $depflag \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

# Preflight: both models, 8 samples/file (~5 min) — validates text-only LLaMA-Guard path.
PF=$(submit grid_guard_preflight 0:30:00 "" \
  "python code/judge_grid_guards.py --judge both --limit 8 --out_dir results/grid_guard_eval/_preflight")

# Full judges run in PARALLEL on two H100s; bigger batches for throughput.
LG=$(submit grid_guard_llamaguard 4:00:00 "$PF" \
  "python code/judge_grid_guards.py --judge llamaguard --llamaguard_batch 16 --out_dir results/grid_guard_eval")
WG_J=$(submit grid_guard_wildguard 4:00:00 "$PF" \
  "python code/judge_grid_guards.py --judge wildguard --wildguard_batch 32 --out_dir results/grid_guard_eval")

# Report after BOTH judges succeed.
REP=$(submit grid_guard_report 0:20:00 "$LG:$WG_J" \
  "python code/report_grid_guards.py results/grid_guard_eval")

echo "──────────────────────────────────────────────────────────────"
echo "preflight=$PF  ->  (llamaguard=$LG ∥ wildguard=$WG_J)  ->  report=$REP"
echo "Watch:   squeue -u \$USER | grep grid_guard"
echo "Preflight log:  tail -f logs/grid_guard_preflight.log"
echo "Final table:    cat logs/grid_guard_report.log   (also results/grid_guard_eval/asr_summary.json)"
echo "──────────────────────────────────────────────────────────────"
