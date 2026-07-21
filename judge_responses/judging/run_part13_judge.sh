#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Part 13 (E1) — Qwen3-VL-8B Instruct vs Thinking under corruption: HR judging.
# Same GPT-4o R/C judge + cache as parts 4/8, so these rows are directly
# comparable to the existing SIUO rows. 8 cells = 2 models x
# {clean, zoom_blur, snow, glass_blur}.
#
# READ FIRST — two things this run depends on:
#   1. OPENAI credits must be live. The judge now FAILS LOUDLY after 3 retries
#      instead of silently scoring 'unharmful' (that silent path is what produced
#      the six 0.0,0.0 R1 rows in part8_ext_hr_summary.csv). A mid-run abort is
#      safe: the cache resumes from the last good item.
#   2. Qwen3-VL-Thinking emits </think> with NO opening <think> (the template puts
#      the opening tag in the prompt). extract_reasoning() has an added branch for
#      exactly this; without it HR_R would be scored over the full response and the
#      reasoning-vs-conclusion comparison would be meaningless. Verified on the
#      real outputs: instruct open=0/close=0, thinking open=0/close~165.
#
# NOTE ON THE INSTRUCT ROWS: with no <think> block at all, extract_reasoning and
# extract_conclusion both fall back to the full response, so instruct HR_R == HR_C
# BY CONSTRUCTION. Report it as a single HR — it is not a reasoning/conclusion split.
#
# WHERE TO RUN: Newton login node, AFTER all 8 inference jobs finish
# (check: every ~/experiments/part13/results/siuo_*_responses.jsonl is 167 lines).
#   conda activate REU
#   export OPENAI_API_KEY=sk-...
#   nohup bash ~/llava_cot_eval/judge_responses/judging/run_part13_judge.sh \
#         > ~/judging/p13_judge.log 2>&1 &
# Numbers -> ~/judging/results_part13/part13_hr_summary.csv (model, HR_R, HR_C).
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R13="$HOME/experiments/part13/results"
OUT="$HOME/judging/results_part13"

python "$HERE/eval_hr_table1_siuo_jsonl_R_C.py" \
  --results-dir "$OUT" \
  --summary-out "$OUT/part13_hr_summary.csv" \
  --entry "clean_qwen3_vl_instruct"       "$R13/siuo_clean_qwen3_vl_instruct_responses.jsonl" \
  --entry "clean_qwen3_vl_thinking"       "$R13/siuo_clean_qwen3_vl_thinking_responses.jsonl" \
  --entry "zoom_blur_qwen3_vl_instruct"   "$R13/siuo_zoom_blur_qwen3_vl_instruct_responses.jsonl" \
  --entry "zoom_blur_qwen3_vl_thinking"   "$R13/siuo_zoom_blur_qwen3_vl_thinking_responses.jsonl" \
  --entry "snow_qwen3_vl_instruct"        "$R13/siuo_snow_qwen3_vl_instruct_responses.jsonl" \
  --entry "snow_qwen3_vl_thinking"        "$R13/siuo_snow_qwen3_vl_thinking_responses.jsonl" \
  --entry "glass_blur_qwen3_vl_instruct"  "$R13/siuo_glass_blur_qwen3_vl_instruct_responses.jsonl" \
  --entry "glass_blur_qwen3_vl_thinking"  "$R13/siuo_glass_blur_qwen3_vl_thinking_responses.jsonl"
