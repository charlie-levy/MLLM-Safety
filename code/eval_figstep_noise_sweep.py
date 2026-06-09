#!/usr/bin/env python
"""
eval_figstep_noise_sweep.py
FigStep ASR sweep over gaussian_noise severity 1-5, for BASE and BASE+TIS.
Results written to results/figstep_noise_sweep/asr_<model>_severity<N>.json
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_figstep
from evaluator import Evaluator
from metrics import compute_asr

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis", action="store_true", help="Load TIS adapter")
parser.add_argument("--severity", type=int, required=True, choices=[1,2,3,4,5])
parser.add_argument("--noise_type", type=str, default="gaussian_noise",
                    choices=["gaussian_noise", "gaussian_blur"])
args = parser.parse_args()

model_tag = "base_tis" if args.use_tis else "base"

print("=" * 80)
print("  FigStep | model=%s | noise=%s | severity=%d" % (model_tag, args.noise_type, args.severity))
print("=" * 80)

print("\n[1/3] Loading FigStep...")
samples = load_figstep()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference...")
evaluator = Evaluator(model, processor,
                      corruption_type=args.noise_type,
                      corruption_severity=args.severity)
results = evaluator.run(samples)

metrics = compute_asr(results)
print("\n" + "=" * 80)
print("FigStep ASR (%s, %s sev=%d): %.2f%% (%d/%d)" % (
    model_tag, args.noise_type, args.severity,
    metrics["asr_pct"], metrics["n_successful"], metrics["n_total"]))
print("=" * 80)

out_dir = "results/figstep_noise_sweep"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "asr_%s_%s_sev%d.json" % (model_tag, args.noise_type, args.severity))
with open(out_file, "w") as f:
    json.dump({
        "model":       model_tag,
        "noise_type":  args.noise_type,
        "severity":    args.severity,
        "asr_pct":     metrics["asr_pct"],
        "n_successful": metrics["n_successful"],
        "n_refused":   metrics["n_refused"],
        "n_total":     metrics["n_total"],
    }, f, indent=2)
print("Saved: %s" % out_file)
