#!/usr/bin/env python
"""
eval_sqa_clean.py — ScienceQA accuracy with no corruption (clean baseline).
Saves in the same format as eval_sqa_noise_sweep.py for consistent plotting.
    results/sqa_noise_sweep/acc_<model>_clean.json
Usage:
  python code/eval_sqa_clean.py              # BASE model
  python code/eval_sqa_clean.py --use_tis    # BASE + TIS
"""
import json, sys, os, argparse
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from evaluator import Evaluator
from metrics import compute_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis", action="store_true", help="Load TIS LoRA adapter")
args = parser.parse_args()

model_tag = "base_tis" if args.use_tis else "base"

print("=" * 80)
print("  ScienceQA Clean Baseline | model=%s | no corruption" % model_tag)
print("=" * 80)

print("\n[1/3] Loading ScienceQA...")
with open("datasets/scienceqa_250.json") as f:
    data = json.load(f)
samples = []
for key in sorted(data.keys(), key=lambda x: int(x)):
    item = data[key]
    samples.append({
        "prompt":   item["prompt"],
        "image":    Image.open(item["image_path"]).convert("RGB"),
        "label":    item["label"],
        "metadata": item,
    })
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference (no corruption)...")
evaluator = Evaluator(model, processor, corruption_type=None)
results = evaluator.run(samples)

metrics = compute_accuracy(results)

print("\n" + "=" * 80)
print("SQA Accuracy (%s, clean): %.2f%% (%d/%d, unknown=%d)" % (
    model_tag, metrics["accuracy"], metrics["n_correct"], metrics["n_total"], metrics["n_unknown"]))
print("=" * 80)

out_dir = "results/sqa_noise_sweep"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "acc_%s_clean.json" % model_tag)
with open(out_file, "w") as f:
    json.dump({
        "model":      model_tag,
        "noise_type": "clean",
        "severity":   0,
        "accuracy":   metrics["accuracy"],
        "correct":    metrics["n_correct"],
        "unknown":    metrics["n_unknown"],
        "total":      metrics["n_total"],
    }, f, indent=2)
print("Saved: %s" % out_file)
