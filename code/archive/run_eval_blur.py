# -*- coding: utf-8 -*-
"""
run_eval_blur.py — BASE + TIS model evaluation on FigStep with blur (severity=3)
"""

import os
import json
import csv
import datetime
import random
import numpy as np
from PIL import Image

from model_loader import load_model_and_processor
from dataset_loader import load_figstep
from evaluator import Evaluator
from metrics import compute_asr, is_refusal
from config import RESULTS_DIR


def save_sample_images(samples: list[dict], model_tag: str, num_samples: int = 5) -> None:
    """Save 5 random before/after blurred images for visual verification."""
    try:
        from imagecorruptions import corrupt

        results_dir = os.path.join(RESULTS_DIR, f"{model_tag}_blur")
        images_dir = os.path.join(results_dir, "sample_images")
        os.makedirs(images_dir, exist_ok=True)

        # Select 5 random samples with images
        valid_samples = [s for s in samples if s.get("image") is not None]
        selected = random.sample(valid_samples, min(num_samples, len(valid_samples)))

        for i, sample in enumerate(selected):
            original_img = sample["image"]
            original_array = np.array(original_img)

            # Apply blur
            blurred_array = corrupt(original_array, corruption_name="gaussian_blur", severity=3)
            blurred_img = Image.fromarray(blurred_array)

            # Save side-by-side
            combined_width = original_img.width + blurred_img.width + 10
            combined = Image.new("RGB", (combined_width, original_img.height), color=(255, 255, 255))
            combined.paste(original_img, (0, 0))
            combined.paste(blurred_img, (original_img.width + 10, 0))

            sample_path = os.path.join(images_dir, f"sample_{i:02d}_before_after.png")
            combined.save(sample_path)
            print(f"    Saved: {sample_path}")

    except Exception as e:
        print(f"    [WARN] Could not save sample images: {e}")


def save_results(tag: str, model_tag: str, results: list[dict], metrics: dict) -> str:
    """Save results to results/<model_tag>_blur/ directory."""
    results_dir = os.path.join(RESULTS_DIR, f"{model_tag}_blur")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = os.path.join(results_dir, f"{tag}_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "label", "is_refusal", "prompt", "response", "metadata"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dataset": r["metadata"].get("dataset", tag),
                "label": r["label"],
                "is_refusal": is_refusal(r["response"]),
                "prompt": r["prompt"],
                "response": r["response"],
                "metadata": json.dumps(r["metadata"]),
            })

    # JSON
    json_path = os.path.join(results_dir, f"{tag}_{timestamp}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"    CSV:     {csv_path}")
    print(f"    Metrics: {json_path}")
    return results_dir


def main():
    print("=" * 70)
    print("  LLaVA-CoT ROBUSTNESS EVALUATION")
    print("=" * 70)
    print("  ✓ Model:      BASE + TIS (exact code with .merge_and_unload())")
    print("  ✓ Dataset:    FigStep only (500 samples)")
    print("  ✓ Corruption: gaussian_blur, severity=3/5")
    print("  ✓ Metric:     ASR (Attack Success Rate)")
    print("=" * 70)

    # Load BASE + TIS model
    print("\n" + "=" * 70)
    print("  STEP 1 — Loading BASE + TIS Model")
    print("=" * 70)
    model, processor, model_tag = load_model_and_processor(use_tis=True)

    # Create evaluator with gaussian_blur corruption
    evaluator = Evaluator(model, processor, corruption_type="gaussian_blur", corruption_severity=3)

    # Load FigStep only
    print("\n" + "=" * 70)
    print("  STEP 2 — Loading Datasets")
    print("=" * 70)
    figstep_samples = load_figstep()

    # Save sample before/after images
    print("\n" + "=" * 70)
    print("  STEP 2.5 — Saving Sample Images (before/after blur)")
    print("=" * 70)
    save_sample_images(figstep_samples, model_tag, num_samples=5)

    # Run inference
    print("\n" + "=" * 70)
    print("  STEP 3 — Running Inference (with corruption)")
    print("=" * 70)
    print(f"\n--- FigStep ({len(figstep_samples)} samples) with gaussian_blur severity=3 ---")
    figstep_results = evaluator.run(figstep_samples)

    # Compute metrics
    print("\n" + "=" * 70)
    print("  STEP 4 — Computing Metrics")
    print("=" * 70)
    figstep_metrics = compute_asr(figstep_results)

    print(f"\n  FigStep ASR:  {figstep_metrics['asr_pct']:.2f}%  ({figstep_metrics['n_successful']}/{figstep_metrics['n_total']} attacks succeeded)")

    # Save results
    print("\n" + "=" * 70)
    print("  STEP 5 — Saving Results")
    print("=" * 70)
    save_results("figstep_blur_severity3", model_tag, figstep_results, figstep_metrics)

    print(f"\nResults saved to: {os.path.join(RESULTS_DIR, f'{model_tag}_blur')}/")


if __name__ == "__main__":
    main()