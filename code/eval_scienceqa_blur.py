#!/usr/bin/env python
"""
Evaluate 250 ScienceQA samples with gaussian_blur corruption.
"""

import json
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from evaluator import Evaluator
from PIL import Image

def load_scienceqa():
    """Load 250 ScienceQA samples."""
    with open('datasets/scienceqa_250.json') as f:
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
    return samples

print("=" * 80)
print("  ScienceQA (250 samples, Base+TIS, BLUR corruption)")
print("=" * 80)

print("\n[1/3] Loading ScienceQA...")
samples = load_scienceqa()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading LLaVA-CoT (Base+TIS)...")
result = load_model_and_processor(use_tis=True)
model, processor = result[0], result[1] if isinstance(result, tuple) else result
print("      OK: base+TIS")

print("\n[3/3] Running inference with BLUR...")
evaluator = Evaluator(model, processor, corruption_type="gaussian_blur", corruption_severity=3)
results = evaluator.run(samples)

print("\n[4/4] Computing accuracy...")
correct = 0
for r in results:
    response = r['response'].lower()
    label = r['label'].lower()
    if label in response:
        correct += 1

accuracy = (correct / len(results)) * 100

print("\nAccuracy with BLUR: %.2f%% (%d/%d)" % (accuracy, correct, len(results)))
print("=" * 80)

os.makedirs("results/scienceqa_blur", exist_ok=True)
with open("results/scienceqa_blur/responses_blur.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results/scienceqa_blur/accuracy.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=2)

print("\nSaved: results/scienceqa_blur/")
print("=" * 80)
