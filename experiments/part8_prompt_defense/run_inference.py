#!/usr/bin/env python
"""
run_inference.py — Part 8 (Experiment 8) INFERENCE driver: PROMPT-BASED DEFENSE
on SIUO. Same models / corruptions / frozen greedy decode as Part 4, but with a
safety instruction PREPENDED TO THE USER PROMPT. Generates responses ONLY — NO
judging (the SIUO HR_R / HR_C judge is run externally, exactly like Parts 4–7).

Idea: can a training-free safety instruction recover the safety that image
corruption breaks? Variants (see prompts.py):
    none       no instruction            (== Part 4 baseline; reuse those files)
    safety     generic safety, no blur mention
    blur_safe  safety + "the image may be blurry, stay cautious"   <- the idea
    perceive   safety + "if you can't see it clearly, don't guess"

NOTE: the instruction is folded into the USER turn, NOT a system message —
Llama-3.2-Vision forbids system messages when an image is present. This routes
through the EXACT frozen Part-4 generate paths, UNCHANGED:
    * Llama family (llava_cot, base_llama) -> code/run_eval.py         (2048 tok)
    * llamav_o1  (Mllama fine-tune)        -> part4/llamav_models.py    (2048 tok)
    * Qwen family (qwen2_5_vl, r1_onevision[_nothink]) -> part4/qwen_models.py (4096 tok)
The ONLY change vs Part 4 is the safety text prepended to the prompt; decoding is
unchanged (do_sample=False, same max_new_tokens). For --prompt none the prompt is
untouched, so that cell reproduces Part 4 exactly.

Output:  siuo_<condition>_<model>_<prompt>_responses.jsonl   (JSONL, per-idx resume)

  python run_inference.py --model llava_cot --condition clean --prompt blur_safe --debug_n 1
  python run_inference.py --model llava_cot --condition zoom_blur --prompt blur_safe
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                    # .../llava_cot_eval
sys.path.insert(0, HERE)                                         # prompts
sys.path.insert(0, os.path.join(REPO, "code"))                   # run_eval, dataset_loader
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))   # qwen_models, llamav_models
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))  # corruption_lib

# importing run_eval chdir's to REPO root and wires up the tested Llama loaders
import run_eval as RE                                          # noqa: E402
from dataset_loader import load_new_attack                     # noqa: E402
from corruption_lib import apply_corruption, severity_for, is_perception_failure  # noqa: E402
from prompts import SYSTEM_PROMPTS                             # noqa: E402

LLAMA_MODELS = ["llava_cot", "base_llama"]
LLAMAV_MODELS = ["llamav_o1"]
QWEN_MODELS = ["qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]
ALL_MODELS = LLAMA_MODELS + LLAMAV_MODELS + QWEN_MODELS
CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
QWEN_MAX_NEW_TOKENS = 4096


# The safety instruction is delivered by PREPENDING it to the user prompt text —
# NOT as a system message. Llama-3.2-Vision's chat template raises
# "Prompting with images is incompatible with system messages" whenever an image
# is present, and every SIUO item has an image. Folding the instruction into the
# user turn is uniform across all models AND routes through the EXACT frozen
# Part-4 generate_one / generate_one_qwen, so no new chat-template path exists to
# break. Semantically identical: the model still sees the safety instruction
# before the question.
def fold_prompt(system, prompt):
    return prompt if system is None else "%s\n\n%s" % (system, prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--prompt", required=True, choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part8/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT every response (nothing written)")
    args = ap.parse_args()

    system = SYSTEM_PROMPTS[args.prompt]
    debug = args.debug_n and args.debug_n > 0
    is_clean = (args.condition == "clean")
    sev = 0 if is_clean else severity_for(args.condition)

    # identity carried into BOTH the filename and the record "model" field, so the
    # external SIUO judge lists each cell distinctly AND lines them up with the other
    # SIUO results. tag == base model for prompt "none" (reproduces the Part-4 filename
    # siuo_<cond>_llava_cot_responses.jsonl exactly), else "<model>_<prompt>".
    tag = args.model if system is None else "%s_%s" % (args.model, args.prompt)
    out_path = os.path.join(args.output_dir, "siuo_%s_%s_responses.jsonl" % (args.condition, tag))
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
    if debug:
        samples = samples[:args.debug_n]

    fam = "Qwen/advisor(4096)" if args.model in QWEN_MODELS else "Llama/run_eval(2048)"
    print("=" * 80, flush=True)
    print("  Part8 INFER | siuo cond=%s(sev%d) model=%s prompt=%s [%s] | %d samples%s  [NO JUDGE]"
          % (args.condition, sev, args.model, args.prompt, fam, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("  system_prompt = %s" % ("<none> (== Part 4)" if system is None else repr(system)), flush=True)
    print("=" * 80, flush=True)

    # Load via the correct frozen path; generation is the UNCHANGED Part-4 path.
    # (The safety instruction is folded into the prompt text by fold_prompt below,
    # so generate() itself is byte-for-byte the Part-4 call.)
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
        if (not is_clean) and image is not None:
            image = apply_corruption(image, args.condition, severity=sev)

        user_prompt = fold_prompt(system, prompt)   # safety instruction prepended (unchanged if none)
        if debug and n_done == 0:
            print("\n---- USER PROMPT SENT TO MODEL (safety instruction folded in) ----\n%s"
                  "\n---- END ----" % user_prompt, flush=True)
        resp = generate(image, user_prompt)

        rec = {
            "idx": idx,
            "model": tag,               # e.g. llava_cot_blur_safe — distinct per prompt, comparable to Part-4 llava_cot
            "base_model": args.model,
            "dataset": "siuo",
            "condition": args.condition,
            "severity": sev,
            "prompt_variant": args.prompt,
            "system_prompt": system,
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
        print("\n[DEBUG] %d responses printed above — confirm the system prompt took effect. Nothing written."
              % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()