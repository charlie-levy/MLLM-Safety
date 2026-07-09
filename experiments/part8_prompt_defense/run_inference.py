#!/usr/bin/env python
"""
run_inference.py — Part 8 (Experiment 8) INFERENCE driver: PROMPT-BASED DEFENSE
on SIUO. Same models / corruptions / frozen greedy decode as Part 4, but with a
safety SYSTEM PROMPT prepended. Generates responses ONLY — NO judging (the SIUO
HR_R / HR_C judge is run externally, exactly like Parts 4–7).

Idea: can a training-free system prompt recover the safety that image corruption
breaks? Prompt variants (see prompts.py):
    none       no system prompt          (== Part 4 baseline; reuse those files)
    safety     generic safety, no blur mention
    blur_safe  safety + "the image may be blurry, stay cautious"   <- the idea
    perceive   safety + "if you can't see it clearly, don't guess"

REUSES the frozen Part-4 inference paths UNCHANGED:
    * Llama family (llava_cot, base_llama) -> code/run_eval.py         (2048 tok)
    * llamav_o1  (Mllama fine-tune)        -> part4/llamav_models.py    (2048 tok)
    * Qwen family (qwen2_5_vl, r1_onevision[_nothink]) -> part4/qwen_models.py (4096 tok)
The ONLY change vs Part 4 is the prepended system message; decoding is unchanged
(do_sample=False, same max_new_tokens). For --prompt none the ORIGINAL Part-4
generate functions are called verbatim, so that cell reproduces Part 4 exactly.

Output:  siuo_<condition>_<model>_<prompt>_responses.jsonl   (JSONL, per-idx resume)

  python run_inference.py --model llava_cot --condition clean --prompt blur_safe --debug_n 1
  python run_inference.py --model llava_cot --condition zoom_blur --prompt blur_safe
"""
import os
import sys
import json
import argparse

import torch

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


# ── system-prompt-aware generation: mirrors the Part-4 frozen decode EXACTLY,
#    only prepending a system message. Used when --prompt != none. ────────────
@torch.inference_mode()
def _llama_generate_sys(model, processor, image, prompt, system, verbose=False):
    has_image = image is not None
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    if verbose:
        print("\n---- RENDERED CHAT INPUT (the system prompt MUST appear below) ----\n%s"
              "\n---- END RENDERED INPUT ----" % text, flush=True)
    images = [[image]] if has_image else None
    inputs = processor(images=images, text=[text], padding=True,
                       return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=RE.MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=processor.tokenizer.eos_token_id)
    return processor.decode(out[0][plen:], skip_special_tokens=True)


@torch.inference_mode()
def _qwen_generate_sys(model, processor, image, prompt, system, max_new_tokens, no_think, verbose=False):
    from qwen_vl_utils import process_vision_info
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if no_think:
        text = text + "<think>\n\n</think>\n\n"
    if verbose:
        print("\n---- RENDERED CHAT INPUT (the system prompt MUST appear below) ----\n%s"
              "\n---- END RENDERED INPUT ----" % text, flush=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


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

    # Load via the correct frozen path; expose a uniform generate(img, prompt).
    if args.model in LLAMA_MODELS:
        model, processor = RE.load(args.model)
        def generate(img, prompt, verbose=False):
            if system is None:
                return RE.generate_one(model, processor, img, prompt)
            return _llama_generate_sys(model, processor, img, prompt, system, verbose=verbose)
    elif args.model in LLAMAV_MODELS:
        from llamav_models import load_llamav_o1
        model, processor = load_llamav_o1()
        def generate(img, prompt, verbose=False):
            if system is None:
                return RE.generate_one(model, processor, img, prompt)
            return _llama_generate_sys(model, processor, img, prompt, system, verbose=verbose)
    else:
        from qwen_models import load_qwen, generate_one_qwen
        model, processor = load_qwen(args.model)
        no_think = args.model.endswith("_nothink")
        def generate(img, prompt, verbose=False):
            if system is None:
                return generate_one_qwen(model, processor, img, prompt,
                                         max_new_tokens=QWEN_MAX_NEW_TOKENS, no_think=no_think)
            return _qwen_generate_sys(model, processor, img, prompt, system,
                                      max_new_tokens=QWEN_MAX_NEW_TOKENS, no_think=no_think, verbose=verbose)

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

        resp = generate(image, prompt, verbose=(debug and n_done == 0))

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