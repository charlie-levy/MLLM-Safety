#!/usr/bin/env python
import json, sys, os
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from evaluator import Evaluator

def load_scienceqa():
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
print("  ScienceQA (250 samples, BASE, CLEAN)")
print("=" * 80)

print("\n[1/3] Loading ScienceQA...")
samples = load_scienceqa()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading LLaVA-CoT (BASE only)...")
result = load_model_and_processor(use_tis=False)
model, processor = result[0], result[1] if isinstance(result, tuple) else result
print("      OK: base")

print("\n[3/3] Running inference...")
evaluator = Evaluator(model, processor, corruption_type=None)
results = evaluator.run(samples)

correct = sum(1 for r in results if r['label'].lower() in r['response'].lower())
accuracy = (correct / len(results)) * 100

print("\n" + "=" * 80)
print("SQA Accuracy (BASE): %.2f%% (%d/%d)" % (accuracy, correct, len(results)))
print("=" * 80)

os.makedirs("results/sqa_base", exist_ok=True)
with open("results/sqa_base/accuracy.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=2)
