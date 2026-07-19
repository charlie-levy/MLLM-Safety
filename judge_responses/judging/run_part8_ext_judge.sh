#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Part 8 EXTENSION (prompt defense on qwen2_5_vl + r1_onevision) — HR judging.
# Same GPT-4o R/C judge + cache as the original Part 8 run; 16 new cells =
# 2 models x {clean, zoom_blur, snow, glass_blur} x {safety, blur_safe}.
# The no-prompt baselines are the Part 4 cells (already judged).
#
# WHERE TO RUN: Newton login node, AFTER all 16 inference jobs finish
# (check: wc -l ~/experiments/part8/results/siuo_*qwen2_5_vl*_responses.jsonl
#         and *r1_onevision* — every file must be 167 lines).
#   conda activate REU
#   nohup bash ~/llava_cot_eval/judge_responses/judging/run_part8_ext_judge.sh \
#         > ~/judging/p8_ext_judge.log 2>&1 &
# Numbers -> ~/judging/results_part8/part8_ext_hr_summary.csv (model, HR_R, HR_C).
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R8="$HOME/experiments/part8/results"
OUT="$HOME/judging/results_part8"

python "$HERE/eval_hr_table1_siuo_jsonl_R_C.py" \
  --results-dir "$OUT" \
  --summary-out "$OUT/part8_ext_hr_summary.csv" \
  --entry "clean_qwen2_5_vl_safety"          "$R8/siuo_clean_qwen2_5_vl_safety_responses.jsonl" \
  --entry "clean_qwen2_5_vl_blur_safe"       "$R8/siuo_clean_qwen2_5_vl_blur_safe_responses.jsonl" \
  --entry "zoom_blur_qwen2_5_vl_safety"      "$R8/siuo_zoom_blur_qwen2_5_vl_safety_responses.jsonl" \
  --entry "zoom_blur_qwen2_5_vl_blur_safe"   "$R8/siuo_zoom_blur_qwen2_5_vl_blur_safe_responses.jsonl" \
  --entry "snow_qwen2_5_vl_safety"           "$R8/siuo_snow_qwen2_5_vl_safety_responses.jsonl" \
  --entry "snow_qwen2_5_vl_blur_safe"        "$R8/siuo_snow_qwen2_5_vl_blur_safe_responses.jsonl" \
  --entry "glass_blur_qwen2_5_vl_safety"     "$R8/siuo_glass_blur_qwen2_5_vl_safety_responses.jsonl" \
  --entry "glass_blur_qwen2_5_vl_blur_safe"  "$R8/siuo_glass_blur_qwen2_5_vl_blur_safe_responses.jsonl" \
  --entry "clean_r1_onevision_safety"          "$R8/siuo_clean_r1_onevision_safety_responses.jsonl" \
  --entry "clean_r1_onevision_blur_safe"       "$R8/siuo_clean_r1_onevision_blur_safe_responses.jsonl" \
  --entry "zoom_blur_r1_onevision_safety"      "$R8/siuo_zoom_blur_r1_onevision_safety_responses.jsonl" \
  --entry "zoom_blur_r1_onevision_blur_safe"   "$R8/siuo_zoom_blur_r1_onevision_blur_safe_responses.jsonl" \
  --entry "snow_r1_onevision_safety"           "$R8/siuo_snow_r1_onevision_safety_responses.jsonl" \
  --entry "snow_r1_onevision_blur_safe"        "$R8/siuo_snow_r1_onevision_blur_safe_responses.jsonl" \
  --entry "glass_blur_r1_onevision_safety"     "$R8/siuo_glass_blur_r1_onevision_safety_responses.jsonl" \
  --entry "glass_blur_r1_onevision_blur_safe"  "$R8/siuo_glass_blur_r1_onevision_blur_safe_responses.jsonl"
