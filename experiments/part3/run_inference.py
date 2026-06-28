#!/usr/bin/env python
"""
run_inference.py — Part 3 INFERENCE driver: LLaVA-CoT+TIS on a materialized Part 2
dataset (MM-SafetyBench-Tiny / VLS-Bench / HoliSafe) under ONE ImageNet-C
corruption. Generates responses ONLY — NO judging (you score these separately with
your own judge program). Same corruptions/severities/model as Part 1.

Reuses the repo's tested pieces UNCHANGED:
  * model loading + frozen generation  -> code/run_eval.py  (RE.load / RE.generate_one;
    bf16, greedy, max_new_tokens=2048, one sample at a time)
  * corruptions + severities           -> experiments/common/corruption_lib.py
    (PART1_CORRUPTIONS, severity_for: 3 blurs at sev5, rest sev3, custom JPEG q5)

Reads the OFFLINE manifest Part 2 already materialized (no HF calls at job time)
and applies the corruption to the clean image before generation.

Output file:  <dataset>_<corruption>_tis_responses.jsonl   (responses-only)
  fields: idx, dataset, corruption, severity, category, prompt, response,
          image_path (clean source), perception_failure
JSONL append + per-idx resume: re-running skips already-written idx.

  python run_inference.py --dataset mmsafety_tiny --corruption fog
  python run_inference.py --dataset vls_bench --corruption glass_blur --debug_n 3
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))

# importing run_eval chdir's to REPO root and wires up the tested loaders
import run_eval as RE                                          # noqa: E402
from corruption_lib import (apply_corruption, PART1_CORRUPTIONS, severity_for,  # noqa: E402
                            is_perception_failure)
from PIL import Image                                          # noqa: E402

MODEL = "llava_cot_tis"   # Part 3 is TIS-only (matches Part 1)
DATA_ROOT = "/home/ch169788/experiments/part2/data"   # reuse Part 2's materialized manifests
DATASETS = ["mmsafety_tiny", "vls_bench", "holisafe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--corruption", required=True, choices=PART1_CORRUPTIONS)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part3/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT every response")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    sev = severity_for(args.corruption)

    manifest = os.path.join(args.data_root, args.dataset, "manifest.jsonl")
    if not os.path.exists(manifest):
        sys.exit("missing manifest %s — run part2/prepare_datasets.py --dataset %s first"
                 % (manifest, args.dataset))
    recs = [json.loads(l) for l in open(manifest) if l.strip()]
    if debug:
        recs = recs[:args.debug_n]

    out_path = os.path.join(args.output_dir, "%s_%s_tis_responses.jsonl" % (args.dataset, args.corruption))
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    # resume: collect already-written idx
    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    print("=" * 78, flush=True)
    print("  Part3 INFER | dataset=%s corruption=%s(sev%d) model=%s | %d samples%s  [NO JUDGE]"
          % (args.dataset, args.corruption, sev, MODEL, len(recs),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 78, flush=True)

    model, processor = RE.load(MODEL)

    n_done = 0
    for r in recs:
        idx = r["idx"]
        if idx in written:
            continue
        prompt = r["prompt"]
        image = Image.open(r["image_path"]).convert("RGB")
        image = apply_corruption(image, args.corruption, severity=sev)

        resp = RE.generate_one(model, processor, image, prompt)

        rec = {
            "idx": idx,
            "dataset": args.dataset,
            "corruption": args.corruption,
            "severity": sev,
            "category": r.get("category", ""),
            "prompt": prompt,
            "response": resp,
            "image_path": r["image_path"],         # clean source; corruption applied in-memory
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
        print("\n[DEBUG] %d responses printed above — confirm they look right. Nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
