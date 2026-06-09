#!/usr/bin/env python
"""
eval_sqa_noise_sweep.py
ScienceQA accuracy sweep over noise/blur severity 1-5, for BASE and BASE+TIS.
Results written to results/sqa_noise_sweep/acc_<model>_<noise>_sev<N>.json
"""
import json, sys, os, argparse
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis", action="store_true", help="Load TIS adapter")
parser.add_argument("--severity", type=int, required=True, choices=[1,2,3,4,5])
parser.add_argument("--noise_type", type=str, default="gaussian_noise",
                    choices=["gaussian_noise", "gaussian_blur"])
args = parser.parse_args()

model_tag = "base_tis" if args.use_tis else "base"

print("=" * 80)
print("  ScienceQA | model=%s | noise=%s | severity=%d" % (model_tag, args.noise_type, args.severity))
print("=" * 80)

print("\n[1/3] Loading ScienceQA...")
with open("datasets/scienceqa_250.json") as f:
    data = json.load(f)
samples = []
for key in sorted(data.keys(), key=lambda x: int(x)):
    item = data[key]
    samples.append({
        "prompt": item["prompt"],
        "image": Image.open(item["image_path"]).convert("RGB"),
        "label": item["label"],
        "metadata": item,
    })
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference...")
evaluator = Evaluator(model, processor,
                      corruption_type=args.noise_type,
                      corruption_severity=args.severity)
results = evaluator.run(samples)

correct = sum(1 for r in results if r["label"].lower() in r["response"].lower())
accuracy = (correct / len(results)) * 100

print("\n" + "=" * 80)
print("SQA Accuracy (%s, %s sev=%d): %.2f%% (%d/%d)" % (
    model_tag, args.noise_type, args.severity, accuracy, correct, len(results)))
print("=" * 80)

out_dir = "results/sqa_noise_sweep"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "acc_%s_%s_sev%d.json" % (model_tag, args.noise_type, args.severity))
with open(out_file, "w") as f:
    json.dump({
        "model": model_tag,
        "noise_type": args.noise_type,
        "severity": args.severity,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
    }, f, indent=2)
print("Saved: %s" % out_file)
