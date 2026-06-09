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
from metrics import compute_asr

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis", action="store_true", help="Load TIS LoRA adapter")
args = parser.parse_args()

model_tag = "base_tis" if args.use_tis else "base"

print("=" * 80)
print("  FigStep Clean Baseline | model=%s | no corruption" % model_tag)
print("=" * 80)

print("\n[1/3] Loading FigStep...")
samples = load_figstep()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
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
