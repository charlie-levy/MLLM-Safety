#!/usr/bin/env python
"""
eval_orr.py — Over-Refusal Rate evaluation on XSTest and MMSA.

Runs inference on both datasets and computes:
  - XSTest ORR (all safe samples)
  - MMSA ORR (indices 0-127)
  - MMSA ORR (indices 2000+)

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
from metrics import compute_orr

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis", action="store_true", help="Load TIS adapter")
args = parser.parse_args()

model_tag = "base_tis" if args.use_tis else "base"

print("=" * 80)
print("  ORR Evaluation (XSTest + MMSA) | model=%s" % model_tag)
print("=" * 80)

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
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
print("\n[4/4] Loading MMSA...")
mmsa_samples = load_mmsa()
print("      OK: %d samples" % len(mmsa_samples))

print("\nRunning MMSA inference...")
mmsa_results = evaluator.run(mmsa_samples)

# Split by index
mmsa_low  = [r for r in mmsa_results if int(r["metadata"]["idx"]) < 128]
mmsa_high = [r for r in mmsa_results if int(r["metadata"]["idx"]) >= 2000]

if not mmsa_low:
    print("[WARN] MMSA subset (idx 0-127) is empty!")
if not mmsa_high:
    print("[WARN] MMSA subset (idx 2000+) is empty!")

mmsa_low_metrics  = compute_orr(mmsa_low,  dataset="mmsa") if mmsa_low  else {}
mmsa_high_metrics = compute_orr(mmsa_high, dataset="mmsa") if mmsa_high else {}

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  ORR Results — model: %s" % model_tag)
print("=" * 80)
print("%-26s| %5s | %8s | %6s" % ("Dataset", "Total", "Refusals", "ORR"))
print("-" * 56)
print("%-26s| %5d | %8d | %5.2f%%" % (
    "XSTest (safe)",
    xstest_metrics["n_total"], xstest_metrics["n_refused"], xstest_metrics["orr_pct"]))
if mmsa_low_metrics:
    print("%-26s| %5d | %8d | %5.2f%%" % (
        "MMSA (idx 0-127)",
        mmsa_low_metrics["n_total"], mmsa_low_metrics["n_refused"], mmsa_low_metrics["orr_pct"]))
if mmsa_high_metrics:
    print("%-26s| %5d | %8d | %5.2f%%" % (
        "MMSA (idx 2000+)",
        mmsa_high_metrics["n_total"], mmsa_high_metrics["n_refused"], mmsa_high_metrics["orr_pct"]))
print("=" * 80)

# ── Save results ──────────────────────────────────────────────────────────────
out_dir = "results/orr_%s" % model_tag
os.makedirs(out_dir, exist_ok=True)
out = {
    "model": model_tag,
    "xstest":    xstest_metrics,
    "mmsa_low":  mmsa_low_metrics,
    "mmsa_high": mmsa_high_metrics,
}
out_file = os.path.join(out_dir, "orr_results.json")
with open(out_file, "w") as f:
    json.dump(out, f, indent=2)
print("Saved: %s" % out_file)
