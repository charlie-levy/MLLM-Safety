#!/usr/bin/env python
"""
run_inference.py — Part 10 INFERENCE driver: BASE LLaVA-CoT (NO TIS) on a Part-2
materialized dataset (MM-SafetyBench-Tiny / SPA-VL / VLS-Bench), under BOTH the
clean and the zoom_blur (severity 2) condition, in ONE model load. Responses ONLY
— judging is done separately, per-paper, by the three GPT-4o judges in this folder.

Why re-run instead of reuse: this is a fresh, self-contained sub-study (base model,
clean vs zoom_blur s=2, judged by each benchmark's OWN protocol). Greedy + seed0 makes
the clean responses reproduce the earlier Part-2 clean responses byte-for-byte, but we
regenerate so the whole study lives in one directory with one provenance.

Reuses the repo's tested pieces UNCHANGED:
  * model + frozen generation -> code/run_eval.py  (RE.load("llava_cot"); bf16, greedy,
    max_new_tokens=2048, seed 0, one sample at a time)   [run_eval frozen contract]
  * corruption               -> experiments/common/corruption_lib.py (apply_corruption)

Reads the OFFLINE Part-2 manifest (no HF calls at job time). For zoom_blur the corruption
is applied to the clean image IN MEMORY right before generation; image_path in the output
always points at the CLEAN source (so the GPT-4o judges see the ground-truth risk image).

Output (one file per condition):  results/<dataset>_<condition>.jsonl
  fields: idx, dataset, condition, corruption, severity, category, prompt, response,
          image_path (clean source), perception_failure
JSONL append + per-idx resume (re-running skips already-written idx).

  python run_inference.py --dataset vls_bench                       # clean + zoom_blur
  python run_inference.py --dataset spa_vl --conditions zoom_blur   # just one
  python run_inference.py --dataset mmsafety_tiny --debug_n 3       # print, write nothing
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))

import run_eval as RE                                          # noqa: E402  (chdir's to REPO)
from corruption_lib import apply_corruption, is_perception_failure  # noqa: E402
from PIL import Image                                          # noqa: E402

MODEL = "llava_cot"                       # BASE LLaVA-CoT, no TIS
DATA_ROOT = "/home/ch169788/experiments/part2/data"           # reuse Part-2 manifests
# holisafe_ssu (SI+ST->U) / holisafe_sss (over-refusal control) are judged by is_refusal
# (HoliSafe SM metric), NOT the GPT-4o judges — same responses-only inference path.
DATASETS = ["mmsafety_tiny", "spa_vl", "vls_bench", "holisafe_ssu", "holisafe_sss"]
ZOOM_SEVERITY = 2                         # the settled zoom_blur severity for this study


def gen_for_condition(model, processor, recs, dataset, condition, out_dir, debug):
    corruption = None if condition == "clean" else "zoom_blur"
    sev = 0 if condition == "clean" else ZOOM_SEVERITY
    out_path = os.path.join(out_dir, "%s_%s.jsonl" % (dataset, condition))

    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    print("\n" + "-" * 78, flush=True)
    print("  condition=%s (corruption=%s sev=%s) -> %s  (%d already done)"
          % (condition, corruption, sev, os.path.basename(out_path), len(written)), flush=True)
    print("-" * 78, flush=True)

    n_done = 0
    for r in recs:
        idx = r["idx"]
        if idx in written:
            continue
        prompt = r["prompt"]
        image = Image.open(r["image_path"]).convert("RGB")
        if corruption is not None:
            image = apply_corruption(image, corruption, severity=sev)

        resp = RE.generate_one(model, processor, image, prompt)

        rec = {
            "idx": idx,
            "dataset": dataset,
            "condition": condition,
            "corruption": corruption or "none",
            "severity": sev,
            "category": r.get("category", ""),
            "prompt": prompt,
            "response": resp,
            "image_path": r["image_path"],        # CLEAN source (judges use this image)
            "perception_failure": is_perception_failure(resp),
        }
        n_done += 1
        if debug:
            print("\n----- idx=%s [%s] cond=%s -----" % (idx, rec["category"], condition))
            print("PROMPT:", prompt[:200])
            print("RESPONSE:\n", resp)
            print("perception_failure=%s" % rec["perception_failure"])
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 25 == 0:
                print("  [%s] %d generated" % (condition, n_done), flush=True)

    print("  DONE condition=%s (%d new)" % (condition, n_done), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--conditions", default="clean,zoom_blur",
                    help="comma list from {clean,zoom_blur} (default both)")
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--output_dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples per condition and PRINT (nothing written)")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        assert c in ("clean", "zoom_blur"), "unknown condition %r" % c
    debug = args.debug_n and args.debug_n > 0

    manifest = os.path.join(args.data_root, args.dataset, "manifest.jsonl")
    if not os.path.exists(manifest):
        sys.exit("missing manifest %s — run part2/prepare_datasets.py --dataset %s first"
                 % (manifest, args.dataset))
    recs = [json.loads(l) for l in open(manifest) if l.strip()]
    if debug:
        recs = recs[:args.debug_n]
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 78, flush=True)
    print("  Part10 INFER | dataset=%s model=%s(base) | conditions=%s | %d samples%s  [NO JUDGE]"
          % (args.dataset, MODEL, conditions, len(recs), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 78, flush=True)

    model, processor = RE.load(MODEL)          # load ONCE, reuse across both conditions
    for cond in conditions:
        gen_for_condition(model, processor, recs, args.dataset, cond, args.output_dir, debug)

    if debug:
        print("\n[DEBUG] responses printed above — confirm they look right. Nothing written.", flush=True)
    else:
        print("\nALL DONE dataset=%s conditions=%s" % (args.dataset, conditions), flush=True)


if __name__ == "__main__":
    main()
