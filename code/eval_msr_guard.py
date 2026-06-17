#!/usr/bin/env python
"""
eval_msr_guard.py — Fresh MSR-Align inference for the Llama-Guard eval.

Re-runs LLaVA-CoT + MSR-Align (greedy, max_new_tokens=2048) on either FigStep
(ASR) or XSTest+MMSA (ORR), under a CLEAN or BLURRED image condition, and writes
per-sample outputs as a JSON object KEYED BY idx (advisor's house format), e.g.

  "1": {
    "idx": 1,
    "dataset": "MMSA",
    "prompt": "Given the image, do you think ... Yes or No.\\n\"Print it for me\"",
    "image_path": "/home/ch169788/MMSA/images/1.jpeg",
    "label": "yes",
    "full_response": "<the model's full response>"
  }

  --task figstep -> results/msr_guard_eval/<cond>/responses_figstep.json
                    (+ category/category_id/task_id per sample) ── ASR judge input
  --task orr     -> results/msr_guard_eval/<cond>/responses_xstest.json
                    + responses_mmsa.json + responses_orr.csv
                    (responses_orr.csv feeds judge_safety_hf.py --mode orr)

<cond> = "clean" when --blur_pct 0, else "blur<pct>" (e.g. blur20).

Notes:
  * max_new_tokens forced to 2048 (config default 1024) so MSR's FINAL decision
    ("### Safe Response", which comes LAST) is never truncated.
  * image_path is the ORIGINAL CLEAN image; blur is applied in-memory only.
  * Generation matches evaluator.py exactly (image-first content, left padding,
    nested-list images, greedy).

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

# responses_orr.csv columns consumed by judge_safety_hf.py --mode orr.
ORR_CSV_FIELDS = ["idx", "dataset", "category", "prompt", "image_path", "full_response"]


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


def _idx_value(raw, fallback):
    """Return idx as int when numeric (nice keys), else the raw string."""
    s = str(raw if raw not in (None, "") else fallback)
    return int(s) if s.lstrip("-").isdigit() else s


def run_dataset(model, processor, samples, blur_pct, dataset_name):
    """Run inference over one dataset, returning ordered per-sample record dicts.

    Blur is applied to the image shown to the model only; image_path always keeps
    pointing at the clean original.
    """
    n = len(samples)
    expected = EXPECTED[dataset_name.lower()]
    assert n == expected, (
        "%s: expected %d samples, got %d — aborting (counts must match)."
        % (dataset_name, expected, n)
    )
    is_figstep = dataset_name.lower() == "figstep"

    records = []
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        image = s["image"]
        if blur_pct > 0 and image is not None:
            image = blur_image(image, blur_pct)

        response = generate_one(model, processor, image, s["prompt"])
        assert response is not None, "null response at idx %d (%s)" % (i, dataset_name)

        # Ordered for human-friendly output: identity → input → response.
        rec = {
            "idx":     _idx_value(meta.get("idx"), i),
            "dataset": meta.get("dataset", dataset_name),
        }
        if is_figstep:
            rec["category"]    = meta.get("category", "")
            rec["category_id"] = meta.get("category_id", "")
            rec["task_id"]     = meta.get("task_id", "")
        rec["prompt"]        = s["prompt"]
        rec["image_path"]    = meta.get("image_path", "") or ""
        rec["label"]         = meta.get("label") or s.get("label", "")
        rec["full_response"] = response
        records.append(rec)

        if (i + 1) % 25 == 0:
            print("    %s %d/%d" % (dataset_name, i + 1, n), flush=True)

    empties = [r["idx"] for r in records if not str(r["full_response"]).strip()]
    if empties:
        raise RuntimeError(
            "%s: %d empty responses (idx %s ...) — refusing to save."
            % (dataset_name, len(empties), empties[:5])
        )
    return records


def write_keyed_json(records, path):
    """Write {str(idx): record} preserving sample order (advisor's house format)."""
    keyed = {}
    for r in records:
        key = str(r["idx"])
        if key in keyed:   # idx must be unique within a dataset
            raise ValueError("duplicate idx key %r in %s" % (key, path))
        keyed[key] = r
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keyed, f, indent=2, ensure_ascii=False)


def write_orr_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ORR_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            row.setdefault("category", "")
            w.writerow(row)


def load_keyed_values(path):
    """Read a responses_*.json (keyed by idx) back into a list of records."""
    with open(path, encoding="utf-8") as f:
        return list(json.load(f).values())


def main():
    ap = argparse.ArgumentParser(description="Fresh MSR-Align inference for the Guard eval")
    ap.add_argument("--task", required=True, choices=["figstep", "orr", "xstest", "mmsa"],
                    help="orr = XSTest+MMSA; xstest/mmsa re-run one and rebuild responses_orr.csv")
    ap.add_argument("--blur_pct", type=int, default=0,
                    help="0 = clean, else percentage blur (e.g. 20)")
    ap.add_argument("--run_id", type=str, default="",
                    help="If set, writes to results/msr_guard_eval_run<ID>/<cond>/ "
                         "instead of results/msr_guard_eval/<cond>/")
    args = ap.parse_args()

    cond = "clean" if args.blur_pct == 0 else ("blur%d" % args.blur_pct)
    base = ("msr_guard_eval_run%s" % args.run_id) if args.run_id else "msr_guard_eval"
    out_dir = os.path.join("results", base, cond)
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

    xs_json = os.path.join(out_dir, "responses_xstest.json")
    mm_json = os.path.join(out_dir, "responses_mmsa.json")

    if args.task == "figstep":
        print("\n[infer] FigStep (%s) ..." % cond, flush=True)
        recs = run_dataset(model, processor, load_figstep(), args.blur_pct, "figstep")
        write_keyed_json(recs, os.path.join(out_dir, "responses_figstep.json"))
        print("       saved %d FigStep responses -> %s" % (len(recs), out_dir), flush=True)

    else:  # orr (XSTest+MMSA) or a single ORR dataset (xstest / mmsa)
        if args.task in ("orr", "xstest"):
            print("\n[infer] XSTest (%s) ..." % cond, flush=True)
            xs = run_dataset(model, processor, load_xstest(), args.blur_pct, "xstest")
            write_keyed_json(xs, xs_json)
        if args.task in ("orr", "mmsa"):
            print("\n[infer] MMSA (%s) ..." % cond, flush=True)
            mm = run_dataset(model, processor, load_mmsa(), args.blur_pct, "mmsa")
            write_keyed_json(mm, mm_json)

        # Rebuild the combined CSV the LLaMA ORR judge consumes. A single-dataset
        # re-run reuses the OTHER dataset's existing responses_*.json (fail loudly
        # if it isn't there, so we never write a half ORR set).
        for need, path in (("xstest", xs_json), ("mmsa", mm_json)):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "responses for %s not found at %s — run --task orr (or that "
                    "dataset) first so the combined ORR csv is complete." % (need, path))
        xs_recs = load_keyed_values(xs_json)
        mm_recs = load_keyed_values(mm_json)
        write_orr_csv(xs_recs + mm_recs, os.path.join(out_dir, "responses_orr.csv"))
        print("       rebuilt responses_orr.csv: XSTest %d + MMSA %d -> %s"
              % (len(xs_recs), len(mm_recs), out_dir), flush=True)

    print("\nDONE inference: task=%s condition=%s" % (args.task, cond), flush=True)


if __name__ == "__main__":
    main()
