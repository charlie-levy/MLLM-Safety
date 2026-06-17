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
parser.add_argument("--use_msr",    action="store_true", help="Load MSR-Align adapter")
parser.add_argument("--use_sage",   action="store_true", help="Load SAGE adapter")
parser.add_argument("--severity",   type=int, default=None, choices=[1,2,3,4,5])
parser.add_argument("--noise_type", type=str, default="gaussian_noise",
                    choices=["gaussian_noise", "gaussian_blur"])
parser.add_argument("--noise_pct", type=int, default=None,
                    help="Percentage noise 0-100 (overrides severity/noise_type)")
parser.add_argument("--blur_pct", type=int, default=None,
                    help="Percentage blur 0-100 (overrides severity/noise_type)")
parser.add_argument("--corrupt", type=str, default=None,
                    choices=["jpeg", "brightness", "pixelate", "motion_blur"],
                    help="Realistic corruption name (use with --corrupt_pct)")
parser.add_argument("--corrupt_pct", type=int, default=None,
                    help="Percentage 0-100 for --corrupt")
args = parser.parse_args()

if args.use_sage:
    model_tag = "base_sage"
elif args.use_msr:
    model_tag = "base_msr"
elif args.use_tis:
    model_tag = "base_tis"
else:
    model_tag = "base"

if args.noise_pct is not None:
    corr_type   = "gaussian_noise_pct"
    corr_sev    = args.noise_pct
    noise_label = "gaussian_noise_pct_p%d" % args.noise_pct
    out_dir     = "results/orr_noise_pct"
elif args.blur_pct is not None:
    corr_type   = "gaussian_blur_pct"
    corr_sev    = args.blur_pct
    noise_label = "gaussian_blur_pct_p%d" % args.blur_pct
    out_dir     = "results/orr_blur_pct"
elif args.corrupt is not None:
    if args.corrupt_pct is None:
        parser.error("--corrupt requires --corrupt_pct")
    corr_type   = "%s_pct" % args.corrupt
    corr_sev    = args.corrupt_pct
    noise_label = "%s_pct_p%d" % (args.corrupt, args.corrupt_pct)
    out_dir     = "results/orr_%s_pct" % args.corrupt
else:
    if args.severity is None:
        parser.error("provide --severity 1-5 or --noise_pct 0-100")
    corr_type   = args.noise_type
    corr_sev    = args.severity
    noise_label = "%s_sev%d" % (args.noise_type, args.severity)
    out_dir     = "results/orr_noise_sweep"

print("=" * 80)
print("  ORR Sweep | model=%s | corruption=%s" % (model_tag, noise_label))
print("=" * 80)

print("\n[1/4] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis, use_msr=args.use_msr, use_sage=args.use_sage)
print("      OK")

evaluator = Evaluator(model, processor,
                      corruption_type=corr_type,
                      corruption_severity=corr_sev)

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

os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "orr_%s_%s.json" % (model_tag, noise_label))
with open(out_file, "w") as f:
    json.dump({
        "model":       model_tag,
        "corruption":  noise_label,
        "noise_pct":   args.noise_pct,
        "blur_pct":    args.blur_pct,
        "severity":    args.severity,
        "xstest":        xstest_metrics,
        "mmsa_combined": mmsa_metrics,
        "avg_orr_pct":   avg_orr_pct,
    }, f, indent=2)
print("Saved: %s" % out_file)

csv_file = os.path.join(out_dir, "responses_%s_%s.csv" % (model_tag, noise_label))
save_results_csv(xstest_results + mmsa_results, csv_file)
print("Saved: %s" % csv_file)
