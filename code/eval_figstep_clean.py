#!/usr/bin/env python
"""
eval_figstep_clean.py — FigStep ASR with no corruption (clean baseline).
Saves in the same format as eval_figstep_noise_sweep.py for consistent plotting.
    results/figstep_noise_sweep/asr_<model>_clean.json
Usage:
  python code/eval_figstep_clean.py              # BASE model
  python code/eval_figstep_clean.py --use_tis    # BASE + TIS
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_figstep
from evaluator import Evaluator
from metrics import compute_asr, save_results_csv

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis",  action="store_true", help="Load TIS LoRA adapter")
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
print("  FigStep Clean Baseline | model=%s | no corruption" % model_tag)
print("=" * 80)

print("\n[1/3] Loading FigStep...")
samples = load_figstep()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis, use_msr=args.use_msr, use_sage=args.use_sage)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference (no corruption)...")
evaluator = Evaluator(model, processor, corruption_type=None)
results = evaluator.run(samples)

metrics = compute_asr(results)

print("\n" + "=" * 80)
print("FigStep ASR (%s, clean): %.2f%% (%d/%d)" % (
    model_tag, metrics["asr_pct"], metrics["n_successful"], metrics["n_total"]))
print("=" * 80)

out_dir = "results/figstep_noise_sweep"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "asr_%s_clean.json" % model_tag)
with open(out_file, "w") as f:
    json.dump({
        "model":      model_tag,
        "noise_type": "clean",
        "severity":   0,
        "asr_pct":    metrics["asr_pct"],
        "n_successful": metrics["n_successful"],
        "n_refused":    metrics["n_refused"],
        "n_total":      metrics["n_total"],
    }, f, indent=2)
print("Saved: %s" % out_file)

csv_file = os.path.join(out_dir, "responses_%s_clean.csv" % model_tag)
save_results_csv(results, csv_file)
print("Saved: %s" % csv_file)
