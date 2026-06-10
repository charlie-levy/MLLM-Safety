#!/usr/bin/env python
"""
eval_orr_noise_sweep.py — ORR (XSTest + MMSA) under image corruption.

Measures how gaussian noise / blur at severities 1-5 affects over-refusal rate
for the BASE model and BASE+TIS. Complements eval_orr.py (clean baseline).

Output: results/orr_noise_sweep/orr_{model}_{noise_type}_sev{N}.json

Usage:
  python code/eval_orr_noise_sweep.py --severity 1 --noise_type gaussian_noise
  python code/eval_orr_noise_sweep.py --severity 3 --noise_type gaussian_blur --use_tis
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_xstest, load_mmsa
from evaluator import Evaluator
from metrics import compute_orr, save_results_csv

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis",    action="store_true", help="Load TIS LoRA adapter")
parser.add_argument("--severity",   type=int, required=True, choices=[1,2,3,4,5])
parser.add_argument("--noise_type", type=str, required=True,
                    choices=["gaussian_noise", "gaussian_blur"])
args = parser.parse_args()

model_tag   = "base_tis" if args.use_tis else "base"
noise_label = "%s_sev%d" % (args.noise_type, args.severity)

print("=" * 80)
print("  ORR Noise Sweep | model=%s | corruption=%s" % (model_tag, noise_label))
print("=" * 80)

print("\n[1/4] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
print("      OK")

evaluator = Evaluator(model, processor,
                      corruption_type=args.noise_type,
                      corruption_severity=args.severity)

print("\n[2/4] Loading XSTest (250 samples)...")
xstest_samples = load_xstest()
print("      OK: %d samples" % len(xstest_samples))

print("\n[3/4] Running XSTest inference (corrupted images)...")
xstest_results = evaluator.run(xstest_samples)
xstest_metrics = compute_orr(xstest_results, dataset="xstest")

print("\n[4/4] Loading + running MMSA (428 safe samples)...")
mmsa_samples = load_mmsa()
print("      OK: %d samples" % len(mmsa_samples))
mmsa_results = evaluator.run(mmsa_samples)
mmsa_metrics = compute_orr(mmsa_results, dataset="mmsa")

avg_orr_pct = round((xstest_metrics["orr_pct"] + mmsa_metrics["orr_pct"]) / 2, 1)

print("\n" + "=" * 80)
print("  ORR Results — %s | %s" % (model_tag, noise_label))
print("=" * 80)
print("%-26s| %5s | %8s | %6s" % ("Dataset", "Total", "Refusals", "ORR"))
print("-" * 56)
print("%-26s| %5d | %8d | %5.1f%%" % (
    "XSTest (250 safe)",
    xstest_metrics["n_total"], xstest_metrics["n_refused"], xstest_metrics["orr_pct"]))
print("%-26s| %5d | %8d | %5.1f%%" % (
    "MMSA (428 safe)",
    mmsa_metrics["n_total"], mmsa_metrics["n_refused"], mmsa_metrics["orr_pct"]))
print("-" * 56)
print("%-26s  %5s   %8s   %5.1f%%" % ("Avg ORR", "", "", avg_orr_pct))
print("=" * 80)

out_dir  = "results/orr_noise_sweep"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "orr_%s_%s.json" % (model_tag, noise_label))
with open(out_file, "w") as f:
    json.dump({
        "model":       model_tag,
        "noise_type":  args.noise_type,
        "severity":    args.severity,
        "xstest":        xstest_metrics,
        "mmsa_combined": mmsa_metrics,
        "avg_orr_pct":   avg_orr_pct,
    }, f, indent=2)
print("Saved: %s" % out_file)

csv_file = os.path.join(out_dir, "responses_%s_%s.csv" % (model_tag, noise_label))
save_results_csv(xstest_results + mmsa_results, csv_file)
print("Saved: %s" % csv_file)
