#!/usr/bin/env python
"""
eval_clean_msr.py — Fresh CLEAN-image inference for MSR-Align (no corruption).

Regenerates full model responses on clean images for:
  • FigStep  (ASR)            -> results/figstep_noise_sweep/responses_base_msr_clean.csv
  • XSTest + MMSA  (ORR)      -> results/orr/responses_base_msr_clean.csv

These CSVs (standard save_results_csv schema, incl. full_response) are then
scored by the LLaMA judge in judge_safety_hf.py. Run on a GPU node.

Usage:
  python code/eval_clean_msr.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_figstep, load_xstest, load_mmsa
from evaluator import Evaluator
from metrics import save_results_csv

print("=" * 80)
print("  CLEAN re-inference | model=base_msr | no corruption")
print("=" * 80)

print("\n[1/4] Loading MSR-Align model ...", flush=True)
model, processor, _ = load_model_and_processor(use_msr=True)
evaluator = Evaluator(model, processor, corruption_type=None)   # None => clean image
print("      OK: base_msr (clean)", flush=True)

# ── FigStep (ASR) ───────────────────────────────────────────────────────────
print("\n[2/4] FigStep inference (clean) ...", flush=True)
fs = load_figstep()
fs_results = evaluator.run(fs)
os.makedirs("results/figstep_noise_sweep", exist_ok=True)
fs_csv = "results/figstep_noise_sweep/responses_base_msr_clean.csv"
save_results_csv(fs_results, fs_csv)
print("      saved %d -> %s" % (len(fs_results), fs_csv), flush=True)

# ── XSTest + MMSA (ORR) ─────────────────────────────────────────────────────
print("\n[3/4] XSTest inference (clean) ...", flush=True)
xs_results = evaluator.run(load_xstest())
print("\n[4/4] MMSA inference (clean) ...", flush=True)
mm_results = evaluator.run(load_mmsa())
os.makedirs("results/orr", exist_ok=True)
orr_csv = "results/orr/responses_base_msr_clean.csv"
save_results_csv(xs_results + mm_results, orr_csv)
print("      saved %d (XSTest %d + MMSA %d) -> %s" % (
    len(xs_results) + len(mm_results), len(xs_results), len(mm_results), orr_csv), flush=True)

print("\nDONE. Clean responses ready for the LLaMA judge.", flush=True)
