#!/usr/bin/env python
"""
eval_orr.py — Over-Refusal Rate evaluation on XSTest and MMSA.

Reproduces Table 2 from the paper:
  ─────────────────────────────────────────────────────
  Model       XSTest ORR   MMSA ORR   Avg ORR   ASR
  ─────────────────────────────────────────────────────
  Base          24.8%        45.0%      34.9%   86.2%
  Base+TIS      56.0%        86.2%      71.1%   11.6%
  ─────────────────────────────────────────────────────

XSTest ORR: 250 safe prompts rendered as text-on-image PNGs.
            Over-refusal = model answers "No" OR uses refusal phrases.

MMSA ORR:   428 safe samples (idx 1-128 + idx 2001-2300).
            Over-refusal = model answers "No" OR uses refusal phrases.

Avg ORR = (XSTest ORR + MMSA ORR) / 2

Usage:
  python code/eval_orr.py              # BASE model
  python code/eval_orr.py --use_tis    # BASE + TIS model
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_xstest, load_mmsa
from evaluator import Evaluator
from metrics import compute_orr, save_results_csv

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis", action="store_true", help="Load TIS LoRA adapter")
parser.add_argument("--use_msr",  action="store_true", help="Load MSR-Align LoRA adapter")
parser.add_argument("--use_sage", action="store_true", help="Load SAGE LoRA adapter")
args = parser.parse_args()

if args.use_sage:
    model_tag = "base_sage"
elif args.use_msr:
    model_tag = "base_msr"
elif args.use_tis:
    model_tag = "base_tis"
else:
    model_tag = "base"

print("=" * 80)
print("  ORR Evaluation (XSTest + MMSA) | model=%s" % model_tag)
print("=" * 80)

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis, use_msr=args.use_msr, use_sage=args.use_sage)
print("      OK: %s" % model_tag)

evaluator = Evaluator(model, processor, corruption_type=None)

# ── XSTest ────────────────────────────────────────────────────────────────────
print("\n[2/4] Loading XSTest...")
xstest_samples = load_xstest()
print("      OK: %d samples" % len(xstest_samples))

print("\n[3/4] Running XSTest inference...")
xstest_results = evaluator.run(xstest_samples)
xstest_metrics = compute_orr(xstest_results, dataset="xstest")

# ── MMSA ──────────────────────────────────────────────────────────────────────
print("\n[4/4] Loading MMSA safe samples (idx 1-128 + 2001-2300)...")
mmsa_samples = load_mmsa()
print("      OK: %d samples" % len(mmsa_samples))

print("\nRunning MMSA inference...")
mmsa_results = evaluator.run(mmsa_samples)

# Combined MMSA ORR (all 428 samples — matches paper's single "MMSA" column)
mmsa_metrics = compute_orr(mmsa_results, dataset="mmsa")

# Per-subset breakdown for reference (idx 1-128 and idx 2001-2300)
mmsa_low  = [r for r in mmsa_results if int(r["metadata"]["idx"]) <= 128]
mmsa_high = [r for r in mmsa_results if int(r["metadata"]["idx"]) >= 2001]

if not mmsa_low:
    print("[WARN] MMSA subset (idx 1-128) is empty — check mmsafeaware_safe.json!")
if not mmsa_high:
    print("[WARN] MMSA subset (idx 2001-2300) is empty — check mmsafeaware_safe.json!")

mmsa_low_metrics  = compute_orr(mmsa_low,  dataset="mmsa") if mmsa_low  else {}
mmsa_high_metrics = compute_orr(mmsa_high, dataset="mmsa") if mmsa_high else {}

# ── Avg ORR (paper Table 2) ───────────────────────────────────────────────────
avg_orr_pct = round((xstest_metrics["orr_pct"] + mmsa_metrics["orr_pct"]) / 2, 1)

# ── Print Table 2-style summary ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("  ORR Results — model: %s  (cf. Table 2 in paper)" % model_tag)
print("=" * 80)
print("%-26s| %5s | %8s | %6s" % ("Dataset", "Total", "Refusals", "ORR"))
print("-" * 56)
print("%-26s| %5d | %8d | %5.1f%%" % (
    "XSTest (250 safe)",
    xstest_metrics["n_total"], xstest_metrics["n_refused"], xstest_metrics["orr_pct"]))
print("%-26s| %5d | %8d | %5.1f%%" % (
    "MMSA (428 safe, combined)",
    mmsa_metrics["n_total"], mmsa_metrics["n_refused"], mmsa_metrics["orr_pct"]))
print("-" * 56)
print("%-26s  %5s   %8s   %5.1f%%" % ("Avg ORR", "", "", avg_orr_pct))
print("=" * 80)
if mmsa_low_metrics:
    print("  MMSA breakdown  idx  1-128 : %5d samples, ORR = %.1f%%" % (
        mmsa_low_metrics["n_total"], mmsa_low_metrics["orr_pct"]))
if mmsa_high_metrics:
    print("  MMSA breakdown  idx 2001+  : %5d samples, ORR = %.1f%%" % (
        mmsa_high_metrics["n_total"], mmsa_high_metrics["orr_pct"]))
print("=" * 80)

# Paper reference values (for easy comparison)
if model_tag == "base":
    print("\n  Paper (Base)     XSTest=24.8  MMSA=45.0  Avg=34.9  ASR=86.2")
elif model_tag == "base_msr":
    print("\n  Paper (MSR-Align) XSTest=57.6  MMSA=76.2  Avg=66.9  ASR=23.8")
else:
    print("\n  Paper (TIS)      XSTest=56.0  MMSA=86.2  Avg=71.1  ASR=11.6")

# ── Save results ──────────────────────────────────────────────────────────────
out_dir = "results/orr_%s" % model_tag
os.makedirs(out_dir, exist_ok=True)
out = {
    "model":         model_tag,
    "xstest":        xstest_metrics,
    "mmsa_combined": mmsa_metrics,
    "avg_orr_pct":   avg_orr_pct,
    "mmsa_low":      mmsa_low_metrics,
    "mmsa_high":     mmsa_high_metrics,
}
out_file = os.path.join(out_dir, "orr_results.json")
with open(out_file, "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: %s" % out_file)

save_results_csv(xstest_results, os.path.join(out_dir, "responses_xstest.csv"))
save_results_csv(mmsa_results,   os.path.join(out_dir, "responses_mmsa.csv"))
print("Saved: %s/responses_xstest.csv + responses_mmsa.csv" % out_dir)
