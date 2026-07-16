#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# HoliSafe SI+ST->U (SSU) Conclusion-ASR judging — SAME GPT-4o R/C judge as
# SIUO / Parts 4/8. 5 models x 4 conditions (clean/zoom_blur/snow/glass_blur).
#
# WHERE TO RUN: Newton login node. API-only: no GPU, no sbatch. `conda activate REU`
# loads OPENAI_API_KEY. Run it backgrounded (stdout is block-buffered to the log):
#   conda activate REU
#   nohup bash ~/judging/run_holisafe_ssu_judge.sh > ~/judging/hs_ssu_judge.log 2>&1 &
#   tail -f ~/judging/hs_ssu_judge.log
# Numbers -> ~/judging/results_holisafe_ssu/holisafe_ssu_conclusion_asr.csv
#   (model = <condition>_<modelkey>, HR_C = Conclusion ASR %). Watch for
#   "[OK] All images found on disk" so it's image-grounded, not text-only.
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/experiments/holisafe_ssu/results"
OUT="$HOME/judging/results_holisafe_ssu"
DS="$HOME/judging/holisafe_ssu_judge_dataset.json"

# 1. Build the judge dataset JSON from the SSU manifest (idempotent).
python ~/llava_cot_eval/experiments/holisafe_ssu/build_judge_dataset.py --out "$DS"

# 2. Judge all cells. --images-dir "" keeps the CLEAN SSU image paths in the JSON
#    (judge scores the response against the true scenario, as in Parts 4/5/8/10).
python "$HERE/eval_hr_table1_siuo_jsonl_R_C.py" \
  --siuo-dataset "$DS" --images-dir "" \
  --results-dir "$OUT" \
  --summary-out "$OUT/holisafe_ssu_conclusion_asr.csv" \
  --entry "clean_base_llama"         "$R/holisafe_ssu_clean_base_llama_responses.jsonl" \
  --entry "glass_blur_base_llama"    "$R/holisafe_ssu_glass_blur_base_llama_responses.jsonl" \
  --entry "snow_base_llama"          "$R/holisafe_ssu_snow_base_llama_responses.jsonl" \
  --entry "zoom_blur_base_llama"     "$R/holisafe_ssu_zoom_blur_base_llama_responses.jsonl" \
  --entry "clean_llava_cot"          "$R/holisafe_ssu_clean_llava_cot_responses.jsonl" \
  --entry "glass_blur_llava_cot"     "$R/holisafe_ssu_glass_blur_llava_cot_responses.jsonl" \
  --entry "snow_llava_cot"           "$R/holisafe_ssu_snow_llava_cot_responses.jsonl" \
  --entry "zoom_blur_llava_cot"      "$R/holisafe_ssu_zoom_blur_llava_cot_responses.jsonl" \
  --entry "clean_llamav_o1"          "$R/holisafe_ssu_clean_llamav_o1_responses.jsonl" \
  --entry "glass_blur_llamav_o1"     "$R/holisafe_ssu_glass_blur_llamav_o1_responses.jsonl" \
  --entry "snow_llamav_o1"           "$R/holisafe_ssu_snow_llamav_o1_responses.jsonl" \
  --entry "zoom_blur_llamav_o1"      "$R/holisafe_ssu_zoom_blur_llamav_o1_responses.jsonl" \
  --entry "clean_qwen2_5_vl"         "$R/holisafe_ssu_clean_qwen2_5_vl_responses.jsonl" \
  --entry "glass_blur_qwen2_5_vl"    "$R/holisafe_ssu_glass_blur_qwen2_5_vl_responses.jsonl" \
  --entry "snow_qwen2_5_vl"          "$R/holisafe_ssu_snow_qwen2_5_vl_responses.jsonl" \
  --entry "zoom_blur_qwen2_5_vl"     "$R/holisafe_ssu_zoom_blur_qwen2_5_vl_responses.jsonl" \
  --entry "clean_r1_onevision"       "$R/holisafe_ssu_clean_r1_onevision_responses.jsonl" \
  --entry "glass_blur_r1_onevision"  "$R/holisafe_ssu_glass_blur_r1_onevision_responses.jsonl" \
  --entry "snow_r1_onevision"        "$R/holisafe_ssu_snow_r1_onevision_responses.jsonl" \
  --entry "zoom_blur_r1_onevision"   "$R/holisafe_ssu_zoom_blur_r1_onevision_responses.jsonl"
