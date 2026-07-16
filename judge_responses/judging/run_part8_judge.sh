#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Part 8 (prompt-based defense on SIUO, model = llava_cot) — Harmful-Rate judging.
# Uses the SIUO Reasoning+Conclusion judge (GPT-4o), SAME judge as Part 4.
#   8 cells = {clean, zoom_blur, snow, glass_blur} x {safety, blur_safe} system prompts.
#
# WHERE TO RUN: inside an interactive `srun` shell (so you can watch it live) — or the
# login node. API-only: no GPU, no sbatch. `conda activate REU` loads your key.
#   srun --cpus-per-task=4 --mem=8G --time=02:00:00 --pty bash
#   conda activate REU
#   bash ~/judging/run_part8_judge.sh          # this is just the plain python call below
# Numbers land in ~/judging/results_part8/part8_hr_summary.csv  (model, HR_R, HR_C).
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R8="$HOME/experiments/part8/results"
OUT="$HOME/judging/results_part8"

python "$HERE/eval_hr_table1_siuo_jsonl_R_C.py" \
  --results-dir "$OUT" \
  --summary-out "$OUT/part8_hr_summary.csv" \
  --entry "clean_safety"         "$R8/siuo_clean_llava_cot_safety_responses.jsonl" \
  --entry "clean_blur_safe"      "$R8/siuo_clean_llava_cot_blur_safe_responses.jsonl" \
  --entry "zoom_blur_safety"     "$R8/siuo_zoom_blur_llava_cot_safety_responses.jsonl" \
  --entry "zoom_blur_blur_safe"  "$R8/siuo_zoom_blur_llava_cot_blur_safe_responses.jsonl" \
  --entry "snow_safety"          "$R8/siuo_snow_llava_cot_safety_responses.jsonl" \
  --entry "snow_blur_safe"       "$R8/siuo_snow_llava_cot_blur_safe_responses.jsonl" \
  --entry "glass_blur_safety"    "$R8/siuo_glass_blur_llava_cot_safety_responses.jsonl" \
  --entry "glass_blur_blur_safe" "$R8/siuo_glass_blur_llava_cot_blur_safe_responses.jsonl"
