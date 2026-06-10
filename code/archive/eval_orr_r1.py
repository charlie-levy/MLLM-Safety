#!/usr/bin/env python
"""
eval_orr_r1.py — ORR evaluation (XSTest + MMSA) for R1-OneVision.
Usage:
  python code/eval_orr_r1.py
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_r1onevision
from dataset_loader import load_xstest, load_mmsa
from evaluator_r1 import R1Evaluator
from metrics import compute_orr

print("=" * 80)
print("  ORR Evaluation (XSTest + MMSA) | model=R1-OneVision")
print("=" * 80)

print("\n[1/4] Loading R1-OneVision...")
model, processor = load_r1onevision()

evaluator = R1Evaluator(model, processor, corruption_type=None)

print("\n[2/4] Loading XSTest...")
xstest_samples = load_xstest()
print("      OK: %d samples" % len(xstest_samples))

print("\n[3/4] Running XSTest inference...")
xstest_results = evaluator.run(xstest_samples)
xstest_metrics = compute_orr(xstest_results, dataset="xstest")

print("\n[4/4] Loading MMSA safe samples (idx 1-128 + 2001-2300)...")
mmsa_samples = load_mmsa()
print("      OK: %d samples" % len(mmsa_samples))

print("\nRunning MMSA inference...")
mmsa_results  = evaluator.run(mmsa_samples)
mmsa_metrics  = compute_orr(mmsa_results, dataset="mmsa")
avg_orr_pct   = round((xstest_metrics["orr_pct"] + mmsa_metrics["orr_pct"]) / 2, 1)

print("\n" + "=" * 80)
print("  ORR Results — R1-OneVision")
print("=" * 80)
print("%-26s| %5d | %8d | %5.1f%%" % (
    "XSTest (250 safe)",
    xstest_metrics["n_total"], xstest_metrics["n_refused"], xstest_metrics["orr_pct"]))
print("%-26s| %5d | %8d | %5.1f%%" % (
    "MMSA (428 safe, combined)",
    mmsa_metrics["n_total"], mmsa_metrics["n_refused"], mmsa_metrics["orr_pct"]))
print("-" * 56)
print("%-26s  %5s   %8s   %5.1f%%" % ("Avg ORR", "", "", avg_orr_pct))
print("=" * 80)

out_dir = "results/orr_r1"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "orr_results.json")
with open(out_file, "w") as f:
    json.dump({
        "model":         "r1onevision",
        "xstest":        xstest_metrics,
        "mmsa_combined": mmsa_metrics,
        "avg_orr_pct":   avg_orr_pct,
    }, f, indent=2)
print("\nSaved: %s" % out_file)
