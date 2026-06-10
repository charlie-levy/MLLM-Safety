#!/usr/bin/env python
"""
debug_tis_sqa.py — Print 5 raw TIS responses for SQA to diagnose extraction failures.
"""
import json, sys, os
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from evaluator import Evaluator
from metrics import extract_answer_letter

with open("datasets/scienceqa_250.json") as f:
    data = json.load(f)

keys = sorted(data.keys(), key=lambda x: int(x))[:5]
samples = [{"prompt": data[k]["prompt"],
            "image": Image.open(data[k]["image_path"]).convert("RGB"),
            "label": data[k]["label"],
            "metadata": data[k]} for k in keys]

print("Loading TIS model...")
model, processor, _ = load_model_and_processor(use_tis=True)
evaluator = Evaluator(model, processor, corruption_type=None)
results = evaluator.run(samples)

for r in results:
    print("=" * 80)
    print("LABEL:    ", r["label"])
    print("EXTRACTED:", extract_answer_letter(r["response"]))
    print("FULL RESPONSE:\n", r["response"])
    print()
