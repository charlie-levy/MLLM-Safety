#!/usr/bin/env python
"""
run_inference.py — Part 12 INFERENCE driver: SIUO DOSE-RESPONSE (severity sweep).
Same models / frozen greedy decode as Part 4; the ONLY new degree of freedom is
an explicit --severity (Part 4 pins severity via severity_for()). Generates
responses ONLY — the SIUO HR_R/HR_C GPT-4o judge runs externally like Parts 4-8.

Why: Table 1 shows single-severity deltas; reviewers will (rightly) ask whether
the effect is monotonic in corruption strength. This sweep gives HR-vs-severity
curves on the HEADLINE benchmark (SIUO) for two model families.

Severity 3 == the Part 4 zoom_blur cell (severity_for("zoom_blur") == 3), so the
sweep reuses that existing run for sev 3 and this driver refuses to duplicate it.

Output:  siuo_<condition>_sev<severity>_<model>_responses.jsonl  (per-idx resume)

  python run_inference.py --model llava_cot --severity 1 --debug_n 1
  python run_inference.py --model qwen2_5_vl --severity 5
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                    # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))                   # run_eval, dataset_loader
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))   # qwen_models, llamav_models
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))  # corruption_lib

# importing run_eval chdir's to REPO root and wires up the tested Llama loaders
import run_eval as RE                                          # noqa: E402
from dataset_loader import load_new_attack                     # noqa: E402
from corruption_lib import apply_corruption, severity_for, is_perception_failure  # noqa: E402

LLAMA_MODELS = ["llava_cot", "base_llama"]
LLAMAV_MODELS = ["llamav_o1"]
QWEN_MODELS = ["qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]
ALL_MODELS = LLAMA_MODELS + LLAMAV_MODELS + QWEN_MODELS
CONDITIONS = ["zoom_blur", "glass_blur", "snow"]
QWEN_MAX_NEW_TOKENS = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--condition", default="zoom_blur", choices=CONDITIONS)
    ap.add_argument("--severity", required=True, type=int, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part12/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT every response (nothing written)")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    if (not debug) and args.severity == severity_for(args.condition):
        sys.exit("severity %d == the Part 4 %s cell; reuse "
                 "~/experiments/part4/results/siuo_%s_%s_responses.jsonl instead of re-running."
                 % (args.severity, args.condition, args.condition, args.model))

    out_path = os.path.join(
        args.output_dir,
        "siuo_%s_sev%d_%s_responses.jsonl" % (args.condition, args.severity, args.model))
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
    assert len(samples) == 167, "SIUO must have 167 items, got %d" % len(samples)
    if debug:
        samples = samples[:args.debug_n]

    fam = "Qwen/advisor(4096)" if args.model in QWEN_MODELS else "Llama/run_eval(2048)"
    print("=" * 80, flush=True)
    print("  Part12 INFER | siuo %s sev=%d model=%s [%s] | %d samples%s  [NO JUDGE]"
          % (args.condition, args.severity, args.model, fam, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    # Load via the correct frozen path; generation is the UNCHANGED Part-4 path.
    if args.model in LLAMA_MODELS:
        model, processor = RE.load(args.model)
        def generate(img, prompt):
            return RE.generate_one(model, processor, img, prompt)
    elif args.model in LLAMAV_MODELS:
        from llamav_models import load_llamav_o1
        model, processor = load_llamav_o1()
        def generate(img, prompt):
            return RE.generate_one(model, processor, img, prompt)
    else:
        from qwen_models import load_qwen, generate_one_qwen
        model, processor = load_qwen(args.model)
        no_think = args.model.endswith("_nothink")
        def generate(img, prompt):
            return generate_one_qwen(model, processor, img, prompt,
                                     max_new_tokens=QWEN_MAX_NEW_TOKENS, no_think=no_think)

    n_done = 0
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        idx = meta.get("idx") or str(i)
        if idx in written:
            continue
        prompt = s["prompt"]
        image = s["image"]
        if image is not None:
            image = apply_corruption(image, args.condition, severity=args.severity)

        resp = generate(image, prompt)

        rec = {
            "idx": idx,
            "model": args.model,
            "dataset": "siuo",
            "condition": args.condition,
            "severity": args.severity,
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
        print("\n[DEBUG] %d responses printed above. Nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
