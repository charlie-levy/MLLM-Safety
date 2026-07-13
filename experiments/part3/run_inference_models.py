#!/usr/bin/env python
"""
run_inference_models.py — Part 3 GENERALIZED: any model on a Part-2 image-safety
dataset (HoliSafe / VLS-Bench / MM-SafetyBench-Tiny) under clean AND/OR corruptions,
mirroring the SIUO reasoning-vs-base setup (Part 4) on these datasets.

Generates responses ONLY (score separately with string-match / your judge). Loads the
model ONCE and loops the requested conditions (Lustre model-load is the expensive part).

Reuses tested paths UNCHANGED:
  Llama family (llava_cot, base_llama, llava_cot_tis) -> code/run_eval.py       (2048 tok)
  Qwen family  (qwen2_5_vl, r1_onevision[_nothink])   -> part4/qwen_models.py    (4096 tok)
Corruptions -> experiments/common/corruption_lib.py (clean = no corruption; same severities).

Output (per condition):  <dataset>_<condition>_<model>_responses.jsonl   (per-idx resume)

  # base LLaVA-CoT on HoliSafe, clean + the 3 SIUO corruptions (one model load):
  python run_inference_models.py --model llava_cot --dataset holisafe \
      --conditions clean,zoom_blur,snow,glass_blur
  # R1-Onevision, same grid:
  python run_inference_models.py --model r1_onevision --dataset holisafe \
      --conditions clean,zoom_blur,snow,glass_blur
  # sanity first (prints 2 responses, writes nothing):
  python run_inference_models.py --model llava_cot --dataset holisafe --conditions zoom_blur --debug_n 2
"""
import os
import sys
import json
import argparse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                     # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))                    # run_eval
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))    # qwen_models
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))   # corruption_lib

import run_eval as RE                                             # noqa: E402  (chdir's to REPO)
from corruption_lib import (apply_corruption, PART1_CORRUPTIONS,   # noqa: E402
                            severity_for, is_perception_failure)

LLAMA_MODELS = ["llava_cot", "base_llama", "llava_cot_tis"]        # code/run_eval.py path
QWEN_MODELS  = ["qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]  # part4/qwen_models path
ALL_MODELS   = LLAMA_MODELS + QWEN_MODELS
DATA_ROOT = "/home/ch169788/experiments/part2/data"               # Part 2's materialized manifests
DATASETS = ["mmsafety_tiny", "vls_bench", "holisafe"]
VALID_CONDITIONS = ["clean"] + list(PART1_CORRUPTIONS)
QWEN_MAX_NEW_TOKENS = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--conditions", required=True,
                    help="comma list from: clean," + ",".join(PART1_CORRUPTIONS))
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part3/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT responses (nothing written)")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        assert c in VALID_CONDITIONS, "unknown condition %r (have %s)" % (c, VALID_CONDITIONS)
    debug = args.debug_n and args.debug_n > 0

    manifest = os.path.join(args.data_root, args.dataset, "manifest.jsonl")
    if not os.path.exists(manifest):
        sys.exit("missing manifest %s — run part2/prepare_datasets.py --dataset %s first"
                 % (manifest, args.dataset))
    recs = [json.loads(l) for l in open(manifest) if l.strip()]
    if debug:
        recs = recs[:args.debug_n]

    fam = "Qwen(4096)" if args.model in QWEN_MODELS else "Llama/run_eval(2048)"
    print("=" * 80, flush=True)
    print("  Part3 GRID INFER | dataset=%s model=%s [%s] | conditions=%s | %d samples%s  [NO JUDGE]"
          % (args.dataset, args.model, fam, conditions, len(recs),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    # load the model ONCE via the correct tested path
    if args.model in LLAMA_MODELS:
        model, processor = RE.load(args.model)
        def generate(img, prompt):
            return RE.generate_one(model, processor, img, prompt)
    else:
        from qwen_models import load_qwen, generate_one_qwen
        model, processor = load_qwen(args.model)
        no_think = args.model.endswith("_nothink")
        def generate(img, prompt):
            return generate_one_qwen(model, processor, img, prompt,
                                     max_new_tokens=QWEN_MAX_NEW_TOKENS, no_think=no_think)

    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    for cond in conditions:
        is_clean = (cond == "clean")
        sev = 0 if is_clean else severity_for(cond)
        out_path = os.path.join(args.output_dir,
                                "%s_%s_%s_responses.jsonl" % (args.dataset, cond, args.model))

        written = set()
        if not debug and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    try:
                        written.add(json.loads(line)["idx"])
                    except Exception:
                        pass

        print("\n----- condition=%s (sev%d) -> %s -----" % (cond, sev, os.path.basename(out_path)), flush=True)
        n_done = 0
        for r in recs:
            idx = r["idx"]
            if idx in written:
                continue
            image = Image.open(r["image_path"]).convert("RGB")
            if not is_clean:
                image = apply_corruption(image, cond, severity=sev)

            resp = generate(image, r["prompt"])

            rec = {
                "idx": idx,
                "dataset": args.dataset,
                "model": args.model,
                "condition": cond,
                "severity": sev,
                "category": r.get("category", ""),
                "prompt": r["prompt"],
                "response": resp,
                "image_path": r["image_path"],       # clean source; corruption applied in-memory
                "perception_failure": is_perception_failure(resp),
            }
            n_done += 1

            if debug:
                print("\n-- idx=%s [%s]" % (idx, rec["category"]))
                print("PROMPT:", r["prompt"][:180])
                print("RESPONSE:\n", resp[:400])
            else:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n_done % 20 == 0:
                    print("  [%s] %d generated" % (cond, n_done), flush=True)

        if debug:
            print("\n[DEBUG] %d responses printed (cond=%s) — nothing written." % (n_done, cond), flush=True)
        else:
            print("DONE %s -> %s  (%d new)" % (cond, out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
