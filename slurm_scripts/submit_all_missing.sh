#!/bin/bash
# ============================================================================
# submit_all_missing.sh — fill every eval gap for the paper table
#
# 53 total jobs across 9 sections:
#
#   A. Base  ASR clean/blur20/blur40  (string-match, eval_figstep_noise_sweep)
#   B. TIS   ASR clean/blur20/blur40  (string-match, eval_figstep_noise_sweep)
#   C. Base  ORR clean                (string-match, eval_orr_noise_sweep)
#   D. TIS   ORR clean                (string-match, eval_orr_noise_sweep)
#   E. Base  SQA blur20/blur40        (LLaMA judge,  eval_sqa_noise_sweep)
#   F. MSR-Align blur40 x 3 runs      (Guard+LLaMA,  eval_msr_guard)
#   G. Llama-3.2-11B-Vision sev 0-5   (string-match, eval_base_vision)
#   H. VLGuard figstep ASR x2 variants x3 cond  (Guard judge, eval_vlguard)
#   I. VLGuard ORR       x2 variants x3 cond    (LLaMA judge, eval_vlguard)
#
# ALREADY VALID — do NOT re-run:
#   VLGuard SQA (mixed=63.6%, posthoc=64.4%):
#     results/vlguard_eval/mixed/clean/judged_vlguard_mixed_sqa.json
#     results/vlguard_eval/posthoc/clean/judged_vlguard_posthoc_sqa.json
#   TIS/MSR/SAGE ORR blur p20-p80: results/orr_blur_pct/
#   TIS/MSR/SAGE SQA blur data:    results/sqa_blur_pct/
#
# DO NOT re-submit:
#   MSR clean+blur20 multirun: already running (jobs 667663-667686)
#
# After all done:
#   python code/report_msr_multirun.py    (now covers clean+blur20+blur40)
#   python code/report_vlguard.py
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

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

COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

submit() {
  local name="$1" tlimit="$2" dep="$3" cmd="$4"
  local depflag=""
  if [ -n "$dep" ]; then depflag="--dependency=afterok:$dep"; fi
  sbatch --parsable $COMMON --job-name="$name" --time="$tlimit" $depflag \
    --output="logs/${name}.log" --wrap="${ENVBLOCK}
${cmd} || exit 1"
}

echo "======================================================================"
echo "  submit_all_missing.sh"
echo "======================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# A. Base ASR clean/blur20/blur40  (string-match)
# Output: results/figstep_blur_pct/asr_base_{clean,gaussian_blur_pct_pN}.json
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- A. Base ASR ---"
submit "base_asr_p0"  6:00:00 "" "python code/eval_figstep_noise_sweep.py --blur_pct 0"
submit "base_asr_p20" 6:00:00 "" "python code/eval_figstep_noise_sweep.py --blur_pct 20"
submit "base_asr_p40" 6:00:00 "" "python code/eval_figstep_noise_sweep.py --blur_pct 40"

# ─────────────────────────────────────────────────────────────────────────────
# B. TIS ASR clean/blur20/blur40  (string-match)
# Output: results/figstep_blur_pct/asr_base+TIS_{clean,gaussian_blur_pct_pN}.json
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- B. TIS ASR ---"
submit "tis_asr_p0"  6:00:00 "" "python code/eval_figstep_noise_sweep.py --use_tis --blur_pct 0"
submit "tis_asr_p20" 6:00:00 "" "python code/eval_figstep_noise_sweep.py --use_tis --blur_pct 20"
submit "tis_asr_p40" 6:00:00 "" "python code/eval_figstep_noise_sweep.py --use_tis --blur_pct 40"

# ─────────────────────────────────────────────────────────────────────────────
# C. Base ORR clean  (string-match)
# Output: results/orr_blur_pct/orr_base_clean.json
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- C. Base ORR clean ---"
submit "base_orr_p0" 6:00:00 "" "python code/eval_orr_noise_sweep.py --blur_pct 0"

# ─────────────────────────────────────────────────────────────────────────────
# D. TIS ORR clean  (string-match)
# Output: results/orr_blur_pct/orr_base+TIS_clean.json
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- D. TIS ORR clean ---"
submit "tis_orr_p0" 6:00:00 "" "python code/eval_orr_noise_sweep.py --use_tis --blur_pct 0"

# ─────────────────────────────────────────────────────────────────────────────
# E. Base SQA blur20/blur40  (inference -> LLaMA judge)
# Inference out: results/sqa_blur_pct/raw_base_gaussian_blur_pct_pN.jsonl
# Judge reads:  results/sqa_blur_pct/ --skip-existing
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- E. Base SQA blur20/blur40 ---"
SQA_P20=$(submit "base_sqa_p20" 3:00:00 "" \
  "python code/eval_sqa_noise_sweep.py --blur_pct 20")
SQA_P40=$(submit "base_sqa_p40" 3:00:00 "" \
  "python code/eval_sqa_noise_sweep.py --blur_pct 40")
submit "base_sqa_judge" 3:00:00 "${SQA_P20}:${SQA_P40}" \
  "python code/judge_sqa_utility_hf.py --dir results/sqa_blur_pct --skip-existing"

# ─────────────────────────────────────────────────────────────────────────────
# F. MSR-Align blur40 x3 independent runs  (Guard ASR + LLaMA ORR)
# Outputs: results/msr_guard_eval_runN/blur40/
#   asr_guard.json        (from judge_figstep_guard.py)
#   judged_llama_orr.json (from judge_safety_hf.py + format_orr_results.py)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- F. MSR blur40 multirun ---"
for R in 1 2 3; do
  OUT="results/msr_guard_eval_run${R}/blur40"

  FIG=$(submit "msrm_fig_r${R}_b40" 8:00:00 "" \
    "python code/eval_msr_guard.py --task figstep --blur_pct 40 --run_id ${R}")

  ORR=$(submit "msrm_orr_r${R}_b40" 8:00:00 "" \
    "python code/eval_msr_guard.py --task orr --blur_pct 40 --run_id ${R}")

  submit "msrm_asr_r${R}_b40" 3:00:00 "$FIG" \
    "python code/judge_figstep_guard.py ${OUT}/responses_figstep.json --out_dir ${OUT}"

  submit "msrm_orrj_r${R}_b40" 3:00:00 "$ORR" \
    "python code/judge_safety_hf.py --mode orr ${OUT}/responses_orr.csv \
     && python code/format_orr_results.py ${OUT}"

  echo "  run${R} blur40: fig_inf=${FIG}  orr_inf=${ORR}"
done

# ─────────────────────────────────────────────────────────────────────────────
# G. Llama-3.2-11B-Vision severity 0-5  (string-match, one job per blur level)
# blur_pct 0/20/40/60/80/100 = severity 0-5
# Output: results/base_vision_eval/blur<N>/metrics.json
# Note: replaces invalid bv_* jobs (667641-667646) which ran with wrong prompt.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- G. Llama-3.2-11B-Vision ---"
for BP in 0 20 40 60 80 100; do
  submit "bv_p${BP}" 8:00:00 "" \
    "python code/eval_base_vision.py --blur_pct ${BP}"
done

# ─────────────────────────────────────────────────────────────────────────────
# H. VLGuard figstep ASR re-run  (Guard judge)
# Covers clean/blur20/blur40 x mixed/posthoc; replaces prior wrong-prompt runs.
# Output: results/vlguard_eval/<variant>/<cond>/responses_figstep.json
#         results/vlguard_eval/<variant>/<cond>/asr_guard.json
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- H. VLGuard figstep ---"
for V in mixed posthoc; do
  for P in 0 20 40; do
    if [ "$P" -eq 0 ]; then COND="clean"; else COND="blur${P}"; fi
    DIR="results/vlguard_eval/${V}/${COND}"

    IF=$(submit "vg_${V}_${COND}_fs" 5:00:00 "" \
      "python code/eval_vlguard.py --variant ${V} --task figstep --blur_pct ${P}")

    submit "vg_${V}_${COND}_asr" 3:00:00 "$IF" \
      "python code/judge_figstep_guard.py ${DIR}/responses_figstep.json --out_dir ${DIR}"

    echo "  vg_${V}_${COND}: inf=${IF}"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# I. VLGuard ORR  (LLaMA-3-8B judge)
# Covers clean/blur20/blur40 x mixed/posthoc.
# Output: results/vlguard_eval/<variant>/<cond>/responses_orr.csv
#         results/vlguard_eval/<variant>/<cond>/judged_llama_orr.json
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- I. VLGuard ORR ---"
for V in mixed posthoc; do
  for P in 0 20 40; do
    if [ "$P" -eq 0 ]; then COND="clean"; else COND="blur${P}"; fi
    DIR="results/vlguard_eval/${V}/${COND}"

    IO=$(submit "vg_${V}_${COND}_orri" 6:00:00 "" \
      "python code/eval_vlguard.py --variant ${V} --task orr --blur_pct ${P}")

    submit "vg_${V}_${COND}_orrj" 3:00:00 "$IO" \
      "python code/judge_safety_hf.py --mode orr ${DIR}/responses_orr.csv \
       && python code/format_orr_results.py ${DIR}"

    echo "  vg_${V}_${COND} orr: inf=${IO}"
  done
done

echo ""
echo "======================================================================"
echo "All 53 gap-fill jobs submitted."
echo ""
echo "Watch:   squeue -u \$USER"
echo ""
echo "When MSR multirun (clean+blur20+blur40) all done:  python code/report_msr_multirun.py"
echo "When VLGuard all done:                             python code/report_vlguard.py"
echo ""
echo "Already valid (no re-run needed):"
echo "  VLGuard SQA: results/vlguard_eval/{mixed,posthoc}/clean/judged_vlguard_*_sqa.json"
echo "  TIS/MSR/SAGE ORR p20-p80: results/orr_blur_pct/"
echo "  TIS/MSR/SAGE SQA blur:    results/sqa_blur_pct/"
echo "======================================================================"
