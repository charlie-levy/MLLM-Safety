#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Part 9 (MSSBench — Multimodal Situational Safety) judging, faithful to the
# official eric-ai-lab/MSSBench GPT-4 evaluator (4-class, text-only).
#   6 models over the 96-item ('if') subset (48 safe/unsafe pairs).
#   llamav_o1 file = the OFFICIAL 4-turn STAGED rerun.
#
# WHERE TO RUN: inside an interactive `srun` shell (so you can watch it live) — or the
# login node. TEXT-ONLY judge: no image, no GPU, no sbatch. `conda activate REU` loads key.
#   srun --cpus-per-task=4 --mem=8G --time=02:00:00 --pty bash
#   conda activate REU
#   bash ~/judging/run_part9_judge.sh          # this is just the plain python call below
# Numbers land in ~/judging/results_part9/mssbench_summary.csv
#   (chat/embodied safe_acc, unsafe_acc, total_acc, overall).
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R9="$HOME/experiments/part9/results"
OUT="$HOME/judging/results_part9"

python "$HERE/judge_mssbench.py" \
  --results-dir "$OUT" \
  --entry llava_cot            "$R9/mss_llava_cot_responses.jsonl" \
  --entry base_llama           "$R9/mss_base_llama_responses.jsonl" \
  --entry llamav_o1            "$R9/mss_llamav_o1_responses.jsonl" \
  --entry qwen2_5_vl           "$R9/mss_qwen2_5_vl_responses.jsonl" \
  --entry r1_onevision         "$R9/mss_r1_onevision_responses.jsonl" \
  --entry r1_onevision_nothink "$R9/mss_r1_onevision_nothink_responses.jsonl"
