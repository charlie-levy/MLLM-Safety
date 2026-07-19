#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Part 12 (SIUO zoom-blur dose-response) — HR judging with the SAME GPT-4o R/C
# judge. 12 entries = 2 models x {clean(sev0), sev1, sev2, sev3, sev4, sev5}.
#   sev1/2/4/5 come from Part 12 inference; sev3 IS the Part 4 zoom_blur cell and
#   clean IS the Part 4 clean cell (re-judged here so we get per-sample verdict
#   files for paired stats — the judge cache makes any overlap cheap).
#
# WHERE TO RUN: Newton login node, AFTER all 8 part12 jobs finish
# (check: wc -l ~/experiments/part12/results/*.jsonl -> eight files x 167).
#   conda activate REU
#   nohup bash ~/llava_cot_eval/judge_responses/judging/run_part12_judge.sh \
#         > ~/judging/p12_judge.log 2>&1 &
# Numbers -> ~/judging/results_part12/part12_dose_response.csv (model, HR_R, HR_C).
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R12="$HOME/experiments/part12/results"
R4="$HOME/experiments/part4/results"
OUT="$HOME/judging/results_part12"

python "$HERE/eval_hr_table1_siuo_jsonl_R_C.py" \
  --results-dir "$OUT" \
  --summary-out "$OUT/part12_dose_response.csv" \
  --entry "zoom_sev0_llava_cot"   "$R4/siuo_clean_llava_cot_responses.jsonl" \
  --entry "zoom_sev1_llava_cot"   "$R12/siuo_zoom_blur_sev1_llava_cot_responses.jsonl" \
  --entry "zoom_sev2_llava_cot"   "$R12/siuo_zoom_blur_sev2_llava_cot_responses.jsonl" \
  --entry "zoom_sev3_llava_cot"   "$R4/siuo_zoom_blur_llava_cot_responses.jsonl" \
  --entry "zoom_sev4_llava_cot"   "$R12/siuo_zoom_blur_sev4_llava_cot_responses.jsonl" \
  --entry "zoom_sev5_llava_cot"   "$R12/siuo_zoom_blur_sev5_llava_cot_responses.jsonl" \
  --entry "zoom_sev0_qwen2_5_vl"  "$R4/siuo_clean_qwen2_5_vl_responses.jsonl" \
  --entry "zoom_sev1_qwen2_5_vl"  "$R12/siuo_zoom_blur_sev1_qwen2_5_vl_responses.jsonl" \
  --entry "zoom_sev2_qwen2_5_vl"  "$R12/siuo_zoom_blur_sev2_qwen2_5_vl_responses.jsonl" \
  --entry "zoom_sev3_qwen2_5_vl"  "$R4/siuo_zoom_blur_qwen2_5_vl_responses.jsonl" \
  --entry "zoom_sev4_qwen2_5_vl"  "$R12/siuo_zoom_blur_sev4_qwen2_5_vl_responses.jsonl" \
  --entry "zoom_sev5_qwen2_5_vl"  "$R12/siuo_zoom_blur_sev5_qwen2_5_vl_responses.jsonl"
