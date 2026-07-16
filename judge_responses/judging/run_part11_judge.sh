#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Part 11 (thinking-budget dose-response on SIUO clean, model = r1_onevision) —
# Harmful-Rate judging. SAME judge as Parts 4/8 (GPT-4o Reasoning+Conclusion).
#   4 rungs = {0, 512, 2048, natural} thinking-token budgets, 50 SIUO items each.
#
# WHERE TO RUN: Newton login node (or an srun shell). API-only: no GPU, no sbatch.
# `conda activate REU` loads OPENAI_API_KEY; images auto-remap via default --images-dir.
#   conda activate REU
#   bash ~/judging/run_part11_judge.sh
# Numbers land in ~/judging/results_part11/part11_dose_response.csv (model, HR_R, HR_C).
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R11="$HOME/experiments/part11/results"
OUT="$HOME/judging/results_part11"

python "$HERE/eval_hr_table1_siuo_jsonl_R_C.py" \
  --results-dir "$OUT" \
  --summary-out "$OUT/part11_dose_response.csv" \
  --entry "budget0"       "$R11/siuo_clean_r1onevision_budget0_responses.jsonl" \
  --entry "budget512"     "$R11/siuo_clean_r1onevision_budget512_responses.jsonl" \
  --entry "budget2048"    "$R11/siuo_clean_r1onevision_budget2048_responses.jsonl" \
  --entry "budgetnatural" "$R11/siuo_clean_r1onevision_budgetnatural_responses.jsonl"
