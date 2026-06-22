#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_base_new_attacks.py — run the NON-REASONING base model
(meta-llama/Llama-3.2-11B-Vision-Instruct, no CoT, no safety training) on the
image-based attack datasets BeaverTails-V (1180) and SIUO (167).

This is the base-Llama counterpart to the LLaVA-CoT BeaverTails grid: the existing
gap_beaver_6_19/run_beavertails.py `--model base` actually loads LLaVA-CoT
(Xkev/Llama-3.2V-11B-cot, the REASONING model). Here we run the genuinely
non-reasoning base model instead — the SAME model and the SAME proven
load/generate path as code/eval_base_vision.py (imported, not re-implemented), so
nothing about the base-model pipeline changes.

  samples  : dataset_loader.load_new_attack(name)  -> the same prepared (image,
             harmful-question) pairs the LLaVA-CoT cells used.
  blur     : blur_utils.blur_image(img, 20)         -> identical to eval_base_vision.
  generate : eval_base_vision.generate_one          -> the EXACT base path
             (apply_chat_template -> processor -> greedy generate, MAX_NEW_TOKENS=512).
  refusal  : metrics.is_refusal (string-match; saved as a convenience field only —
             the deliverable is the FULL responses; judge later if needed).

Perturbation is applied IN-MEMORY per image (originals untouched). Incremental save
every 50 + RESUME (skips already-done idx), so a preempt loses <=50.

  python code/eval_base_new_attacks.py --dataset both        --blur_pct 20
  python code/eval_base_new_attacks.py --dataset beavertails --blur_pct 20
  python code/eval_base_new_attacks.py --dataset siuo        --blur_pct 20 --pilot

Output (keyed by idx, like eval_base_vision):
  results/base_vision_new_attacks/blur20/responses_beavertails.json
  results/base_vision_new_attacks/blur20/responses_siuo.json
  results/base_vision_new_attacks/blur20/metrics.json
Each response entry:
  idx, dataset, prompt, image_path, category, label, full_response, is_refusal
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                            # noqa: E402
from tqdm import tqdm                                   # noqa: E402

# Reuse the EXACT base-model pipeline (load + greedy generate) — do not re-implement.
from eval_base_vision import load_model, generate_one   # noqa: E402
from dataset_loader import load_new_attack              # noqa: E402
from blur_utils import blur_image                       # noqa: E402
from metrics import is_refusal                          # noqa: E402

from config import NEW_ATTACK_COUNTS                    # noqa: E402

SAVE_EVERY = 50


def _load_existing(path):
    """Resume: return (records_list, done_idx_set). Accepts a keyed dict or a list."""
    if not os.path.exists(path):
        return [], set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = list(data.values()) if isinstance(data, dict) else list(data)
    return records, {str(r["idx"]) for r in records}


def _write_keyed(records, path):
    """Write neat JSON keyed by idx (duplicate idx is a hard error)."""
    keyed = {}
    for r in records:
        k = str(r["idx"])
        if k in keyed:
            raise ValueError("duplicate idx %r in %s" % (k, path))
        keyed[k] = r
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keyed, f, indent=2, ensure_ascii=False)


def run_dataset(model, processor, name, blur_pct, out_dir, pilot=False):
    samples = load_new_attack(name)                     # asserts count internally
    if pilot:
        samples = samples[:5]
    n = len(samples)
    out_path = os.path.join(out_dir, "responses_%s.json" % name)

    records, done = _load_existing(out_path)
    if not pilot and all(str(s["metadata"]["idx"]) in done for s in samples):
        nonref = sum(1 for r in records if not r["is_refusal"])
        print("[%s] already complete (%d entries) — skipping." % (name, len(records)))
        print("[%s] non-refusal rate = %.2f%% (%d/%d)  [cached]"
              % (name, 100.0 * nonref / n, nonref, n))
        return out_path, n, nonref

    pbar = tqdm(total=n, initial=len(records), desc="%s/blur%d" % (name, blur_pct), unit="img")
    for i, s in enumerate(samples):
        idx = str(s["metadata"]["idx"])
        if idx in done:
            continue
        img = s["image"]
        if blur_pct > 0 and img is not None:
            img = blur_image(img, blur_pct)             # in-memory; original untouched
        resp = generate_one(model, processor, img, s["prompt"])
        assert resp is not None and resp.strip(), "empty response at %s (%s)" % (idx, name)
        ref = bool(is_refusal(resp))
        records.append({
            "idx":           idx,
            "dataset":       s["metadata"].get("dataset", name),
            "prompt":        s["prompt"],
            "image_path":    s["metadata"].get("image_path", "") or "",
            "category":      s["metadata"].get("category", ""),
            "label":         s.get("label", "harmful"),
            "full_response": resp,
            "is_refusal":    ref,
        })
        done.add(idx)
        pbar.update(1)
        if len(records) % SAVE_EVERY == 0:
            _write_keyed(records, out_path)
    pbar.close()

    _write_keyed(records, out_path)
    if not pilot:
        assert len(records) == n, "%s: expected %d entries, wrote %d" % (name, n, len(records))
    nonref = sum(1 for r in records if not r["is_refusal"])
    print("[%s] saved %d responses -> %s" % (name, len(records), out_path))
    print("[%s] non-refusal rate = %.2f%% (%d/%d)" % (name, 100.0 * nonref / n, nonref, n))
    return out_path, n, nonref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both",
                    choices=["beavertails", "siuo", "both"])
    ap.add_argument("--blur_pct", type=int, default=20,
                    choices=[0, 20, 40, 60, 80, 100],
                    help="0=clean, 20=blur20 (the requested level), ...")
    ap.add_argument("--pilot", action="store_true", help="first 5 images only (smoke test)")
    args = ap.parse_args()

    names = ["beavertails", "siuo"] if args.dataset == "both" else [args.dataset]
    cond = "clean" if args.blur_pct == 0 else ("blur%d" % args.blur_pct)
    out_dir = os.path.join("results", "base_vision_new_attacks", cond)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("  BASE (non-reasoning) Llama-3.2-11B-Vision-Instruct | %s | %s%s"
          % (", ".join(names), cond, "  (PILOT)" if args.pilot else ""))
    print("=" * 80, flush=True)

    model, processor = load_model()                     # the proven base-model loader

    summary = {"model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
               "blur_pct": args.blur_pct, "datasets": {}}
    for name in names:
        out_path, n, nonref = run_dataset(model, processor, name, args.blur_pct,
                                          out_dir, pilot=args.pilot)
        summary["datasets"][name] = {
            "n": n,
            "n_non_refusal": nonref,
            "non_refusal_rate_pct": round(100.0 * nonref / n, 2) if n else 0.0,
            "responses_file": out_path,
            "expected_count": NEW_ATTACK_COUNTS[name],
        }

    if not args.pilot:
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    for name, d in summary["datasets"].items():
        print("  %-12s n=%-5d non-refusal=%.2f%% (%d)"
              % (name, d["n"], d["non_refusal_rate_pct"], d["n_non_refusal"]))
    print("  Saved -> %s" % out_dir)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
