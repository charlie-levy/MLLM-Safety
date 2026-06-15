#!/usr/bin/env python
"""
eval_msr_guard.py — Fresh MSR-Align inference for the Llama-Guard eval.

Re-runs LLaVA-CoT + MSR-Align (greedy, max_new_tokens=2048) on either FigStep
(ASR) or XSTest+MMSA (ORR), under a CLEAN or BLURRED image condition, and writes
rich per-sample outputs that capture EVERYTHING needed downstream:

  --task figstep  -> results/msr_guard_eval/<cond>/responses_figstep.{jsonl,csv}
                     (idx, dataset, category, category_id, task_id, prompt,
                      image_path, full_response)  ── fed to judge_figstep_guard.py
  --task orr      -> results/msr_guard_eval/<cond>/responses_orr.csv
                     (idx, dataset, category, prompt, image_path, full_response)
                     + responses_xstest.jsonl + responses_mmsa.jsonl
                     ── responses_orr.csv is fed to judge_safety_hf.py --mode orr

<cond> = "clean" when --blur_pct 0, else "blur<pct>" (e.g. blur20).

IMPORTANT — why a dedicated driver (not eval_clean_msr.py / the sweep drivers):
  * max_new_tokens is forced to 2048 (config default is 1024) because MSR-Align's
    FINAL decision ("### Safe Response") comes LAST after a long <think> +
    Image Description + Risk Assessment; 1024 can truncate the actual verdict.
  * image_path is preserved per FigStep sample so the Guard judge can later show
    the model the ORIGINAL CLEAN image regardless of the blur condition.

Generation matches evaluator.py exactly (image-first content, left padding,
nested-list images, greedy) so responses are comparable to the rest of the repo.

Run on a Newton H100 GPU node, offline. One (task, blur_pct) per job.

Usage:
  python code/eval_msr_guard.py --task figstep --blur_pct 0
  python code/eval_msr_guard.py --task orr     --blur_pct 20
"""
import os
import sys
import csv
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from model_loader import load_model_and_processor          # noqa: E402
from dataset_loader import load_figstep, load_xstest, load_mmsa  # noqa: E402
from blur_utils import blur_image                            # noqa: E402

MAX_NEW_TOKENS = 2048   # override config.MAX_NEW_TOKENS (=1024): MSR verdict is last
EXPECTED = {"figstep": 500, "xstest": 250, "mmsa": 428}

# Columns shared by every responses_*.csv we emit (superset; FigStep adds two more).
ORR_CSV_FIELDS = ["idx", "dataset", "category", "prompt", "image_path", "full_response"]
FIGSTEP_CSV_FIELDS = ["idx", "dataset", "category", "category_id", "task_id",
                      "prompt", "image_path", "full_response"]


def build_messages(prompt, has_image):
    """Identical content ordering to evaluator.Evaluator._build_messages."""
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def generate_one(model, processor, image, prompt):
    """Single-sample greedy generation, mirroring evaluator._run_batch (BATCH_SIZE=1)."""
    has_image = image is not None
    text = processor.apply_chat_template(
        build_messages(prompt, has_image), add_generation_prompt=True
    )
    images = [[image]] if has_image else None
    inputs = processor(
        images=images, text=[text], padding=True, return_tensors="pt"
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def run_dataset(model, processor, samples, blur_pct, dataset_name):
    """Run inference over one dataset, returning a list of rich record dicts.

    Applies in-memory blur to the image actually shown to the model, but never
    mutates the stored image_path (which keeps pointing at the clean original).
    """
    n = len(samples)
    expected = EXPECTED[dataset_name.lower()]
    assert n == expected, (
        "%s: expected %d samples, got %d — aborting (counts must match)."
        % (dataset_name, expected, n)
    )

    records = []
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        image = s["image"]
        if blur_pct > 0 and image is not None:
            image = blur_image(image, blur_pct)

        response = generate_one(model, processor, image, s["prompt"])
        assert response is not None, "null response at idx %d (%s)" % (i, dataset_name)

        records.append({
            "idx":          meta.get("idx", str(i)),
            "dataset":      meta.get("dataset", dataset_name),
            "category":     meta.get("category", ""),
            "category_id":  meta.get("category_id", ""),
            "task_id":      meta.get("task_id", ""),
            "prompt":       s["prompt"],
            "image_path":   meta.get("image_path", "") or "",
            "full_response": response,
        })

        if (i + 1) % 25 == 0:
            print("    %s %d/%d" % (dataset_name, i + 1, n), flush=True)

    # No silent failures: every row must carry a non-empty response.
    empties = [r["idx"] for r in records if not str(r["full_response"]).strip()]
    if empties:
        raise RuntimeError(
            "%s: %d empty responses (idx %s ...) — refusing to save."
            % (dataset_name, len(empties), empties[:5])
        )
    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(records, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Fresh MSR-Align inference for the Guard eval")
    ap.add_argument("--task", required=True, choices=["figstep", "orr"])
    ap.add_argument("--blur_pct", type=int, default=0,
                    help="0 = clean, else percentage blur (e.g. 20)")
    args = ap.parse_args()

    cond = "clean" if args.blur_pct == 0 else ("blur%d" % args.blur_pct)
    out_dir = os.path.join("results", "msr_guard_eval", cond)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("  MSR-Align inference | task=%s | condition=%s | max_new_tokens=%d"
          % (args.task, cond, MAX_NEW_TOKENS))
    print("=" * 80, flush=True)

    print("\n[load] LLaVA-CoT + MSR-Align ...", flush=True)
    model, processor, tag = load_model_and_processor(use_msr=True)
    processor.tokenizer.padding_side = "left"   # match evaluator (batched generation safe)
    assert tag == "base+MSR", "expected base+MSR, got %r" % tag
    print("       OK: %s" % tag, flush=True)

    if args.task == "figstep":
        print("\n[infer] FigStep (%s) ..." % cond, flush=True)
        recs = run_dataset(model, processor, load_figstep(), args.blur_pct, "figstep")
        write_jsonl(recs, os.path.join(out_dir, "responses_figstep.jsonl"))
        write_csv(recs, os.path.join(out_dir, "responses_figstep.csv"), FIGSTEP_CSV_FIELDS)
        print("       saved %d FigStep responses -> %s" % (len(recs), out_dir), flush=True)

    else:  # orr = XSTest + MMSA
        print("\n[infer] XSTest (%s) ..." % cond, flush=True)
        xs = run_dataset(model, processor, load_xstest(), args.blur_pct, "xstest")
        print("\n[infer] MMSA (%s) ..." % cond, flush=True)
        mm = run_dataset(model, processor, load_mmsa(), args.blur_pct, "mmsa")

        write_jsonl(xs, os.path.join(out_dir, "responses_xstest.jsonl"))
        write_jsonl(mm, os.path.join(out_dir, "responses_mmsa.jsonl"))
        # Combined CSV (dataset column distinguishes XSTest vs MMSA) for the ORR judge.
        write_csv(xs + mm, os.path.join(out_dir, "responses_orr.csv"), ORR_CSV_FIELDS)
        print("       saved %d ORR responses (XSTest %d + MMSA %d) -> %s"
              % (len(xs) + len(mm), len(xs), len(mm), out_dir), flush=True)

    print("\nDONE inference: task=%s condition=%s" % (args.task, cond), flush=True)


if __name__ == "__main__":
    main()
