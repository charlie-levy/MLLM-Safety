"""
dataset_loader.py — Loads and normalizes all three evaluation datasets.

Each loader returns a list of sample dicts with a unified schema so the
Evaluator doesn't need to know which dataset it's processing:

    {
        "prompt":   str,               # text instruction sent to the model
        "image":    PIL.Image | None,  # visual input (None → blank placeholder used)
        "label":    str,               # "harmful" | "safe"  (for metric routing)
        "metadata": dict,              # dataset-specific fields kept for CSV output
    }

Dataset roles
─────────────
  FigStep  →  ASR  (Attack Success Rate)    label = "harmful"
  XSTest   →  ORR  (Over-Refusal Rate)      label = "safe"   (safe subset only)
  MMSA     →  ORR  (Over-Refusal Rate)      label = "safe"   (benign subset only)
"""

import os
import json
import glob
import pandas as pd
from PIL import Image

from config import (
    FIGSTEP_REPO_PATH, FIGSTEP_CSV, FIGSTEP_IMAGE_DIR,
    XSTEST_SAFE_JSON, XSTEST_IMAGE_DIR,
    MMSA_SAFE_JSON, MMSA_IMAGE_DIR, MMSA_IMAGE_DIR2,
)


# ── Utility ────────────────────────────────────────────────────────────────────

def _blank_image(width: int = 336, height: int = 336) -> Image.Image:
    """
    Returns a plain white RGB image used as a placeholder when a dataset has
    no visual input but LLaVA-CoT (a VLM) still requires one.
    336×336 matches LLaVA's default vision encoder tile size.
    """
    return Image.new("RGB", (width, height), color=(255, 255, 255))


# ── FigStep ────────────────────────────────────────────────────────────────────

def load_figstep(split: str = "test") -> list[dict]:
    """
    Loads the FigStep jailbreak benchmark from a locally cloned GitHub repo.

    FigStep evades text-based safety filters by rendering harmful instructions
    as text *inside an image*. The model receives the image and a neutral
    prompt asking it to follow what the image says. A trial counts as an
    'attack success' if the model complies instead of refusing.

    Repo: https://github.com/CryptoAILab/FigStep
    Clone to FIGSTEP_REPO_PATH defined in config.py.

    Returns a list of sample dicts with label = "harmful".
    """
    csv_path   = os.path.join(FIGSTEP_REPO_PATH, FIGSTEP_CSV)
    image_root = os.path.join(FIGSTEP_REPO_PATH, FIGSTEP_IMAGE_DIR)

    print(f"[dataset_loader] Loading FigStep from '{csv_path}' ...")

    df = pd.read_csv(csv_path)

    # Columns: dataset, category_id, task_id, category_name, question, instruction
    samples = []
    for _, row in df.iterrows():
        # Use the natural-language question as the prompt to the model
        prompt = str(row["question"])

        # Images are named: query_{dataset}_{category_id}_{task_id}_*.png
        # e.g. query_ForbidQI_1_3_6.png  (the trailing number is a text-style variant)
        # We use glob to match regardless of the style suffix.
        pattern = os.path.join(
            image_root,
            f"query_{row['dataset']}_{row['category_id']}_{row['task_id']}_*.png"
        )
        matches = glob.glob(pattern)

        if matches:
            image_path = matches[0]
            image = Image.open(image_path).convert("RGB")
        else:
            print(f"[dataset_loader] Warning: no image found for pattern {pattern}, using blank.")
            image_path = None
            image = _blank_image()

        samples.append({
            "prompt":   prompt,
            "image":    image,
            "label":    "harmful",
            "metadata": {
                "dataset":     "FigStep",
                "category":    str(row["category_name"]),
                "category_id": str(row["category_id"]),
                "task_id":     str(row["task_id"]),
                "image_path":  image_path,
            },
        })

    print(f"[dataset_loader] FigStep: {len(samples)} samples loaded.")
    return samples


# ── XSTest ─────────────────────────────────────────────────────────────────────

def load_xstest(split: str = "test") -> list[dict]:
    """
    Loads XSTest safe prompts from the pre-rendered image dataset.

    Each safe XSTest prompt is typeset on a blank PNG image by
    create_xstest_images.py. The model receives the image and a fixed
    prompt asking it to extract the text and assess whether the request
    is safe (Yes/No). ORR = model answers "No" for a safe prompt.

    Run create_xstest_images.py once before evaluating.
    """
    print(f"[dataset_loader] Loading XSTest from '{XSTEST_SAFE_JSON}' ...")

    with open(XSTEST_SAFE_JSON, encoding="utf-8") as f:
        data = json.load(f)

    missing = 0
    samples = []
    for item in data.values():
        idx = int(item["idx"])
        img_path = item["image_path"]

        # If the stored path doesn't exist (e.g. copied from a different user's system),
        # try to find the image in XSTEST_IMAGE_DIR by idx.
        if not os.path.exists(img_path):
            for fname in [f"{idx:04d}.png", f"{idx - 1:04d}.png", f"{idx}.png"]:
                candidate = os.path.join(XSTEST_IMAGE_DIR, fname)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = _blank_image()
            missing += 1

        samples.append({
            "prompt":   item["prompt"],
            "image":    image,
            "label":    "safe",
            "metadata": {
                "dataset":    "XSTest",
                "idx":        str(item["idx"]),
                "image_path": img_path if os.path.exists(img_path) else "",
                "label":      item.get("label", ""),   # raw dataset label, if any
            },
        })

    if missing:
        print(f"[dataset_loader] Warning: {missing} XSTest images missing, used blank.")
    print(f"[dataset_loader] XSTest: {len(samples)} safe samples loaded (250 expected).")
    return samples


# ── MMSA ───────────────────────────────────────────────────────────────────────

def load_mmsa(split: str = "test") -> list[dict]:
    """
    Loads only the SAFE MMSA samples (label=yes, indices 1-128 and 2001-2300).

    The full MMSA dataset contains both safe and unsafe samples; for ORR we
    only evaluate on safe ones. mmsafeaware_safe.json contains exactly these
    428 samples. ORR = model answers "No" (judges safe content as unsafe).

    Images are looked up from the local MMSA image directories by idx.
    """
    print(f"[dataset_loader] Loading MMSA safe samples from '{MMSA_SAFE_JSON}' ...")

    with open(MMSA_SAFE_JSON, encoding="utf-8") as f:
        data = json.load(f)

    missing = 0
    samples = []
    for item in data.values():
        idx = item["idx"]

        image = None
        image_path = ""
        for img_dir in [MMSA_IMAGE_DIR, MMSA_IMAGE_DIR2]:
            for ext in ["jpeg", "jpg", "png"]:
                path = os.path.join(img_dir, f"{idx}.{ext}")
                if os.path.exists(path):
                    image = Image.open(path).convert("RGB")
                    image_path = path
                    break
            if image:
                break

        if image is None:
            image = _blank_image()
            missing += 1

        samples.append({
            "prompt":   item["prompt"],
            "image":    image,
            "label":    "safe",
            "metadata": {
                "dataset":    "MMSA",
                "idx":        str(idx),
                "image_path": image_path,
                "label":      item.get("label", ""),   # raw dataset label (e.g. "yes")
            },
        })

    if missing:
        print(f"[dataset_loader] Warning: {missing} MMSA images missing, used blank.")
    print(f"[dataset_loader] MMSA: {len(samples)} safe samples loaded (428 expected).")
    return samples
