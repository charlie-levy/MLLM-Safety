#!/usr/bin/env python
"""
eval_figstep_r1.py — FigStep ASR sweep for R1-OneVision.
Usage:
  python code/eval_figstep_r1.py                       # clean (no corruption)
  python code/eval_figstep_r1.py --severity 3 --noise_type gaussian_noise
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_r1onevision
from dataset_loader import load_figstep
from evaluator_r1 import R1Evaluator
from metrics import compute_asr

parser = argparse.ArgumentParser()
parser.add_argument("--severity",   type=int, default=0, choices=[0,1,2,3,4,5])
parser.add_argument("--noise_type", type=str, default="clean",
                    choices=["clean", "gaussian_noise", "gaussian_blur"])
args = parser.parse_args()

corruption = None if args.noise_type == "clean" or args.severity == 0 else args.noise_type
noise_label = "clean" if corruption is None else "%s_sev%d" % (args.noise_type, args.severity)

print("=" * 80)
print("  FigStep R1-OneVision | corruption=%s" % noise_label)
print("=" * 80)

print("\n[1/3] Loading FigStep...")
samples = load_figstep()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading R1-OneVision...")
model, processor = load_r1onevision()

print("\n[3/3] Running inference...")
evaluator = R1Evaluator(model, processor,
                        corruption_type=corruption,
                        corruption_severity=args.severity)
results = evaluator.run(samples)

metrics = compute_asr(results)
print("\n" + "=" * 80)
print("FigStep ASR (R1-OneVision, %s): %.2f%%" % (noise_label, metrics["asr_pct"]))
print("=" * 80)

out_dir = "results/figstep_r1"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "asr_%s.json" % noise_label)
with open(out_file, "w") as f:
    json.dump({
        "model":        "r1onevision",
        "noise_type":   args.noise_type,
        "severity":     args.severity,
        "asr_pct":      metrics["asr_pct"],
        "n_successful": metrics["n_successful"],
        "n_refused":    metrics["n_refused"],
        "n_total":      metrics["n_total"],
    }, f, indent=2)
print("Saved: %s" % out_file)
