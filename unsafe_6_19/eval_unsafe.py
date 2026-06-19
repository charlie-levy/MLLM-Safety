#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_unsafe.py — Phase 2/3: generate LLaVA-CoT responses on the unsafe images.

Conditions (model x blur):  base_clean, tis_clean, tis_blur_20, tis_blur_40.
Loads ONE model and can run several blur levels per call (no reload per condition).
Blur is applied IN-MEMORY to a copy; the original files are never modified, and
image_path stays the CLEAN original (so the judge sees the clean image).

  python unsafe_6_19/eval_unsafe.py --blur 0                  # base, clean
  python unsafe_6_19/eval_unsafe.py --use_tis --blur 0 20 40  # TIS, clean+20+40
  python unsafe_6_19/eval_unsafe.py --blur 0 --pilot          # PILOT: image 0 only

INCREMENTAL + RESUMABLE: writes the output file every few images and SKIPS images
already done on restart — so a preempt/bump loses at most a couple images, and a
rerun continues where it stopped.

Output per condition (keyed by image index):
  unsafe_6_19/responses_<condition>[_pilot].json
    { "<idx>": {image_index, image_path(CLEAN), condition, prompt, model, model_response} }
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)   # self-contained: model_loader/blur_utils/config live here

import torch                                       # noqa: E402
from PIL import Image                              # noqa: E402
from model_loader import load_model_and_processor  # noqa: E402
from blur_utils import blur_image                  # noqa: E402

DATASET_JSON = "unsafe_6_19/unsafe_beavertails_v.json"
OUT_DIR = "unsafe_6_19"
MAX_NEW_TOKENS = 1024     # base/TIS standard
SAVE_EVERY = 5            # write progress every N images (resume granularity)


def build_messages(prompt):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


@torch.inference_mode()
def generate_one(model, processor, image, prompt):
    text = processor.apply_chat_template(build_messages(prompt), add_generation_prompt=True)
    inputs = processor(images=[[image]], text=[text], padding=True, return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=processor.tokenizer.eos_token_id)
    return processor.decode(out[0][plen:], skip_special_tokens=True)


def load_samples(pilot):
    data = json.load(open(DATASET_JSON, encoding="utf-8"))
    samples = [data[k] for k in sorted(data, key=lambda x: int(x))]
    if pilot:
        samples = samples[:1]
    for s in samples:
        assert os.path.exists(s["image_path"]), "missing image: %s" % s["image_path"]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_tis", action="store_true", help="base+TIS (else base)")
    ap.add_argument("--blur", type=int, nargs="+", default=[0], help="blur pct(s): 0 20 40")
    ap.add_argument("--pilot", action="store_true", help="only image index 0")
    args = ap.parse_args()

    samples = load_samples(args.pilot)
    n = len(samples)
    mtag = "tis" if args.use_tis else "base"
    suffix = "_pilot" if args.pilot else ""
    print("[eval] %d image(s) | model=%s | blur=%s%s"
          % (n, mtag, args.blur, "  (PILOT)" if args.pilot else ""), flush=True)

    model, processor, tag = load_model_and_processor(use_tis=args.use_tis)
    processor.tokenizer.padding_side = "left"
    print("[eval] loaded %s" % tag, flush=True)

    for pct in args.blur:
        cond = ("%s_clean" % mtag) if pct == 0 else ("%s_blur_%d" % (mtag, pct))
        out_path = os.path.join(OUT_DIR, "responses_%s%s.json" % (cond, suffix))
        recs = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
        done_before = len(recs)

        for i, s in enumerate(samples):
            key = str(s["idx"])
            if key in recs and str(recs[key].get("model_response", "")).strip():
                continue                                   # resume: already done
            img = Image.open(s["image_path"]).convert("RGB")
            shown = blur_image(img, pct) if pct > 0 else img
            resp = generate_one(model, processor, shown, s["prompt"])
            assert resp is not None and resp.strip(), "empty response at idx %s" % key
            recs[key] = {
                "image_index":    s["idx"],
                "image_path":     s["image_path"],         # CLEAN original
                "condition":      cond,
                "prompt":         s["prompt"],
                "model":          mtag,
                "model_response": resp,
            }
            if (i + 1) % SAVE_EVERY == 0:
                json.dump(recs, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                print("  [%s] %d/%d saved" % (cond, i + 1, n), flush=True)

        json.dump(recs, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("[eval] %s -> %s  (%d total, %d new this run)"
              % (cond, out_path, len(recs), len(recs) - done_before), flush=True)


if __name__ == "__main__":
    main()
