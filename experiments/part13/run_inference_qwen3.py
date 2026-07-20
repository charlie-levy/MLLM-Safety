#!/usr/bin/env python
"""
run_inference_qwen3.py — Part 13 INFERENCE driver: SIUO under ONE condition
(clean / zoom_blur / snow / glass_blur) for the Qwen3-VL reasoning-isolation set.
Generates responses ONLY — NO judging (reuse the part4/part8 GPT-4o R/C judge).

Models (see qwen3_vl_models.py):
    qwen3_vl_instruct           reasoning OFF (different weights)
    qwen3_vl_thinking           reasoning ON  (native)
    qwen3_vl_thinking_nothink   reasoning OFF (SAME weights as thinking)  <- isolation

APPLES-TO-APPLES with the existing SIUO results: same samples
(dataset_loader.load_new_attack("siuo")), same corruptions/severities
(corruption_lib: zoom_blur sev3, snow sev3, glass_blur sev5), same greedy
decoding, same 4096-token Qwen-family budget, and the IDENTICAL output jsonl
schema the part4 responses use — so the same judge reads them unchanged.

  python run_inference_qwen3.py --model qwen3_vl_thinking --condition clean --debug_n 2
  python run_inference_qwen3.py --model qwen3_vl_thinking_nothink --condition zoom_blur
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, HERE)                                 # qwen3_vl_models
sys.path.insert(0, os.path.join(REPO, "code"))           # dataset_loader
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))  # corruption_lib

from dataset_loader import load_new_attack                     # noqa: E402
from corruption_lib import apply_corruption, severity_for, is_perception_failure  # noqa: E402
from qwen3_vl_models import load_qwen3, generate_one_qwen3, NOTHINK_KEYS  # noqa: E402

MODELS = ["qwen3_vl_instruct", "qwen3_vl_thinking", "qwen3_vl_thinking_nothink"]
CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
QWEN_MAX_NEW_TOKENS = 4096                       # matches the existing Qwen-family path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part13/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT every response (nothing written)")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    is_clean = (args.condition == "clean")
    sev = 0 if is_clean else severity_for(args.condition)
    no_think = args.model in NOTHINK_KEYS

    out_path = os.path.join(args.output_dir, "siuo_%s_%s_responses.jsonl" % (args.condition, args.model))
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    samples = load_new_attack("siuo")
    assert len(samples) == 167, "SIUO must be 167 items, got %d" % len(samples)
    if debug:
        samples = samples[:args.debug_n]

    print("=" * 80, flush=True)
    print("  Part13 INFER | dataset=siuo condition=%s(sev%d) model=%s no_think=%s | %d samples%s  [NO JUDGE]"
          % (args.condition, sev, args.model, no_think, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    model, processor = load_qwen3(args.model)

    def generate(img, prompt):
        return generate_one_qwen3(model, processor, img, prompt,
                                  max_new_tokens=QWEN_MAX_NEW_TOKENS, no_think=no_think)

    n_done = 0
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        idx = meta.get("idx") or str(i)
        if idx in written:
            continue
        prompt = s["prompt"]
        image = s["image"]
        if (not is_clean) and image is not None:
            image = apply_corruption(image, args.condition, severity=sev)

        resp = generate(image, prompt)
        if not resp:
            raise RuntimeError("EMPTY response at idx=%s (%s/%s) — fail loudly, do not write blanks"
                               % (idx, args.model, args.condition))

        rec = {
            "idx": idx,
            "model": args.model,
            "dataset": "siuo",
            "condition": args.condition,
            "severity": sev,
            "category": meta.get("category", ""),
            "prompt": prompt,
            "response": resp,
            "image_path": meta.get("image_path", ""),
            "perception_failure": is_perception_failure(resp),
        }
        n_done += 1

        if debug:
            print("\n----- idx=%s [%s] -----" % (idx, rec["category"]))
            print("PROMPT:", prompt[:200])
            print("RESPONSE:\n", resp)
            print("perception_failure=%s" % rec["perception_failure"])
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 10 == 0:
                print("  %d generated" % n_done, flush=True)

    if debug:
        print("\n[DEBUG] %d responses printed — CONFIRM the thinking model shows a <think> block "
              "and the nothink model does NOT. Nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
