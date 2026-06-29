#!/usr/bin/env python
"""
run_inference.py — Part 4 (Experiment 4) INFERENCE driver: SIUO under ONE
condition (clean / zoom_blur / snow / glass_blur), for FOUR models forming two
reasoning-vs-base pairs. Generates responses ONLY — NO judging.

Models:
    llava_cot     reasoning   (Xkev/Llama-3.2V-11B-cot)                 } Llama pair
    base_llama    base        (meta-llama/Llama-3.2-11B-Vision-Instruct) }
    r1_onevision  reasoning   (Fancy-MLLM/R1-Onevision-7B)              } Qwen pair
    qwen2_5_vl    base        (Qwen/Qwen2.5-VL-7B-Instruct)             }

Apples-to-apples: same SIUO samples, same corruption (corruption_lib), greedy
decoding for all. Loading/generation REUSE the established paths UNCHANGED:
    * Llama pair -> code/run_eval.py  (RE.load / RE.generate_one; 2048 tok)
    * Qwen pair  -> part4/qwen_models.py (advisor's tested code; 4096 tok)
Corruptions -> experiments/common/corruption_lib.py (clean = no corruption;
zoom_blur sev3, snow sev3, glass_blur sev5 — same severities as Parts 1/3).
SIUO loader -> code/dataset_loader.load_new_attack("siuo").

Output file:  siuo_<condition>_<model>_responses.jsonl
JSONL append + per-idx resume.

  python run_inference.py --model llava_cot    --condition clean
  python run_inference.py --model r1_onevision --condition glass_blur --debug_n 2
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, HERE)                                 # qwen_models
sys.path.insert(0, os.path.join(REPO, "code"))           # run_eval, dataset_loader
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))  # corruption_lib

# importing run_eval chdir's to REPO root and wires up the tested Llama loaders
import run_eval as RE                                          # noqa: E402
from dataset_loader import load_new_attack                     # noqa: E402
from corruption_lib import apply_corruption, severity_for, is_perception_failure  # noqa: E402

LLAMA_MODELS = ["llava_cot", "base_llama"]      # existing run_eval path (UNCHANGED, 2048 tok)
QWEN_MODELS = ["qwen2_5_vl", "r1_onevision"]    # new qwen_models path (advisor's code, 4096 tok)
ALL_MODELS = LLAMA_MODELS + QWEN_MODELS
CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
QWEN_MAX_NEW_TOKENS = 4096                       # Llama pair uses run_eval's frozen 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part4/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT every response")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    is_clean = (args.condition == "clean")
    sev = 0 if is_clean else severity_for(args.condition)

    out_path = os.path.join(args.output_dir, "siuo_%s_%s_responses.jsonl" % (args.condition, args.model))
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

    samples = load_new_attack("siuo")
    if debug:
        samples = samples[:args.debug_n]

    fam = "Llama/run_eval(2048)" if args.model in LLAMA_MODELS else "Qwen/advisor(4096)"
    print("=" * 80, flush=True)
    print("  Part4 INFER | dataset=siuo condition=%s(sev%d) model=%s [%s] | %d samples%s  [NO JUDGE]"
          % (args.condition, sev, args.model, fam, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    # Load via the correct path; expose a uniform generate(img, prompt) closure.
    if args.model in LLAMA_MODELS:
        model, processor = RE.load(args.model)                          # existing, UNCHANGED
        def generate(img, prompt):
            return RE.generate_one(model, processor, img, prompt)
    else:
        from qwen_models import load_qwen, generate_one_qwen
        model, processor = load_qwen(args.model)
        def generate(img, prompt):
            return generate_one_qwen(model, processor, img, prompt, max_new_tokens=QWEN_MAX_NEW_TOKENS)

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
        print("\n[DEBUG] %d responses printed above — confirm they look right. Nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
