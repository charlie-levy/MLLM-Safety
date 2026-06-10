#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ScienceQA evaluation - inference only (no judging). Results saved for local judgment."""

import json
import sys
import os
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

from model_loader import load_model_and_processor
from evaluator import Evaluator
from metrics import is_refusal

SCIENCEQA_PATH = os.path.join(PROJECT_DIR, 'datasets', 'scienceqa_250.json')


def load_scienceqa():
    """Load ScienceQA dataset with images."""
    if not os.path.exists(SCIENCEQA_PATH):
        print("[data] NOT FOUND")
        return []

    with open(SCIENCEQA_PATH) as f:
        data = json.load(f)

    samples = []
    for key, item in data.items():
        img_path = item.get("image_path", "")
        image = None

        if img_path:
            full_path = os.path.join(PROJECT_DIR, img_path)
            if os.path.exists(full_path):
                try:
                    image = Image.open(full_path).convert("RGB")
                except Exception as e:
                    print("[warn] Failed to load image %s: %s" % (full_path, str(e)))

        samples.append({
            "idx": item.get("idx", key),
            "prompt": item.get("prompt", ""),
            "image": image,
            "label": item.get("label", ""),
            "metadata": {"idx": item.get("idx", key), "dataset": "ScienceQA"}
        })

    return samples


def main():
    print("\n" + "="*80)
    print("  SCIENCEQA INFERENCE ONLY (Judging will be local)")
    print("="*80)

    print("\n[1/3] Loading ScienceQA...")
    samples = load_scienceqa()
    if not samples:
        return
    print("      OK: %d samples" % len(samples))

    print("\n[2/3] Loading LLaVA-CoT (Base+TIS)...")
    try:
        model, processor, tag = load_model_and_processor(use_tis=True)
        print("      OK: %s" % tag)
    except Exception as e:
        print("      ERROR: %s" % str(e))
        return

    print("\n[3/3] Running inference...")
    try:
        evaluator = Evaluator(model, processor)
        results = evaluator.run(samples)
        print("      OK: %d responses" % len(results))
    except Exception as e:
        print("      ERROR: %s" % str(e))
        return

    responses = []
    for i, r in enumerate(results):
        responses.append({
            "idx": r["metadata"].get("idx"),
            "prompt": r["prompt"],
            "response": r["response"],
            "label": r["label"],
            "is_refusal": is_refusal(r["response"])
        })

        if (i + 1) % 50 == 0:
            print("      [%d/%d] responses saved" % (i + 1, len(results)))

        if (i + 1) % 25 == 0:
            print("      [%d/%d] saved" % (i + 1, len(results)))

    outdir = "results/scienceqa"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, "responses_raw.json")

    with open(outpath, "w") as f:
        json.dump(responses, f, indent=2)

    print("\n" + "="*80)
    print("  INFERENCE COMPLETE")
    print("="*80)
    print("  Responses: %d" % len(responses))
    print("  Saved: %s" % outpath)
    print("  Next: Run judge_responses_local.py locally")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print("\nERROR: %s" % str(e))
        import traceback
        traceback.print_exc()
