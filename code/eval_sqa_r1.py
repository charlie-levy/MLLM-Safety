#!/usr/bin/env python
"""
eval_sqa_r1.py — ScienceQA accuracy sweep for R1-OneVision.
Usage:
  python code/eval_sqa_r1.py                       # clean
  python code/eval_sqa_r1.py --severity 3 --noise_type gaussian_noise
"""
import json, sys, os, argparse
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_r1onevision
from evaluator_r1 import R1Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--severity",   type=int, default=0, choices=[0,1,2,3,4,5])
parser.add_argument("--noise_type", type=str, default="clean",
                    choices=["clean", "gaussian_noise", "gaussian_blur"])
args = parser.parse_args()

corruption = None if args.noise_type == "clean" or args.severity == 0 else args.noise_type
noise_label = "clean" if corruption is None else "%s_sev%d" % (args.noise_type, args.severity)

print("=" * 80)
print("  ScienceQA R1-OneVision | corruption=%s" % noise_label)
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

print("\n[2/3] Loading R1-OneVision...")
model, processor = load_r1onevision()

print("\n[3/3] Running inference...")
evaluator = R1Evaluator(model, processor,
                        corruption_type=corruption,
                        corruption_severity=args.severity)
results = evaluator.run(samples)

correct  = sum(1 for r in results if r["label"].lower() in r["response"].lower())
accuracy = (correct / len(results)) * 100

print("\n" + "=" * 80)
print("SQA Accuracy (R1-OneVision, %s): %.2f%% (%d/%d)" % (
    noise_label, accuracy, correct, len(results)))
print("=" * 80)

out_dir = "results/sqa_r1"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "acc_%s.json" % noise_label)
with open(out_file, "w") as f:
    json.dump({
        "model":      "r1onevision",
        "noise_type": args.noise_type,
        "severity":   args.severity,
        "accuracy":   accuracy,
        "correct":    correct,
        "total":      len(results),
    }, f, indent=2)
print("Saved: %s" % out_file)
