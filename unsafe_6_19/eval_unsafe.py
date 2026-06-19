#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_unsafe.py — generate LLaVA-CoT responses on the 100 unsafe BeaverTails-V images.

For each image: show the model the image (optionally blurred) + the single prompt
from unsafe_beavertails_v.json, and save the FULL response. Loads ONE model and can
do several blur levels in a single run (so we don't reload the model per condition).

  python unsafe_6_19/eval_unsafe.py --blur 0                 # base, clean
  python unsafe_6_19/eval_unsafe.py --use_tis --blur 0 20 40 # TIS, clean+blur20+blur40
  python unsafe_6_19/eval_unsafe.py --use_tis --blur 0 --limit 1   # 1-image TEST

Output (one file per blur level), keyed by idx:
  unsafe_6_19/responses_<base|tis>_<clean|blurNN>.json
    { "<idx>": {idx, dataset, image_path(CLEAN), prompt, model, condition, full_response} }

image_path stays the CLEAN original (blur is applied in-memory only) so the judge
later sees the clean image. Greedy decoding (deterministic). Reuses the repo's
tested model_loader + blur_utils — no model code is changed.
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)   # self-contained: model_loader.py + blur_utils.py + config.py live here

import torch                                       # noqa: E402
from PIL import Image                              # noqa: E402
from model_loader import load_model_and_processor  # noqa: E402
from blur_utils import blur_image                  # noqa: E402

DATASET_JSON = "unsafe_6_19/unsafe_beavertails_v.json"
OUT_DIR = "unsafe_6_19"
MAX_NEW_TOKENS = 1024   # base/TIS standard (not MSR, which would need 2048)


def build_messages(prompt):
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]


@torch.inference_mode()
def generate_one(model, processor, image, prompt):
    """Single-sample greedy generation, mirroring the proven eval_msr_guard path."""
    text = processor.apply_chat_template(build_messages(prompt), add_generation_prompt=True)
    inputs = processor(images=[[image]], text=[text], padding=True, return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=processor.tokenizer.eos_token_id)
    return processor.decode(out[0][plen:], skip_special_tokens=True)


def load_samples(limit):
    with open(DATASET_JSON, encoding="utf-8") as f:
        data = json.load(f)
    samples = [data[k] for k in sorted(data, key=lambda x: int(x))]
    if limit:
        samples = samples[:limit]
    for s in samples:                              # fail loudly if any image is missing
        assert os.path.exists(s["image_path"]), "missing image: %s" % s["image_path"]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_tis", action="store_true", help="load base+TIS (else base)")
    ap.add_argument("--blur", type=int, nargs="+", default=[0],
                    help="blur pct level(s); 0=clean, e.g. --blur 0 20 40")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (TEST)")
    args = ap.parse_args()

    samples = load_samples(args.limit)
    n = len(samples)
    print("[eval] %d images | blur levels=%s | model=%s"
          % (n, args.blur, "tis" if args.use_tis else "base"), flush=True)

    model, processor, tag = load_model_and_processor(use_tis=args.use_tis)
    processor.tokenizer.padding_side = "left"
    mtag = "tis" if args.use_tis else "base"
    print("[eval] loaded %s" % tag, flush=True)

    for pct in args.blur:
        cond = "clean" if pct == 0 else ("blur%d" % pct)
        recs = {}
        for i, s in enumerate(samples):
            img = Image.open(s["image_path"]).convert("RGB")
            shown = blur_image(img, pct) if pct > 0 else img
            resp = generate_one(model, processor, shown, s["prompt"])
            assert resp is not None, "null response at idx %s" % s["idx"]
            recs[str(s["idx"])] = {
                "idx":           s["idx"],
                "dataset":       s.get("dataset", "BeaverTails-V"),
                "image_path":    s["image_path"],   # CLEAN original
                "prompt":        s["prompt"],
                "model":         mtag,
                "condition":     cond,
                "full_response": resp,
            }
            if (i + 1) % 10 == 0:
                print("  [%s/%s] %d/%d" % (mtag, cond, i + 1, n), flush=True)

        empties = [k for k, v in recs.items() if not str(v["full_response"]).strip()]
        if empties:
            raise RuntimeError("%s/%s: empty responses idx %s" % (mtag, cond, empties[:5]))

        out = os.path.join(OUT_DIR, "responses_%s_%s.json" % (mtag, cond))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        print("[eval] wrote %s  (%d responses)" % (out, len(recs)), flush=True)


if __name__ == "__main__":
    main()
