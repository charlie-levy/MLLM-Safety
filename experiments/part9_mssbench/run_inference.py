#!/usr/bin/env python
"""
run_inference.py — Part 9 (MSSBench) INFERENCE driver. Runs one of our models over
the class-stratified MSSBench instruction-following ('if') subset produced by
prepare_subset.py. Generates responses ONLY — judging (GPT-4o safe/unsafe per the
paper) is done externally.

Protocol (eric-ai-lab/MSSBench, exact): each manifest item is a single image + the
text `PROMPT_{CHAT,EMBODIED}_IF + query/instruction`. safe/unsafe contexts are
separate items sharing a pair_id. Uses the FROZEN Part-4 generate paths UNCHANGED:
    * Llama family (llava_cot, base_llama) -> code/run_eval.py         (2048 tok)
    * llamav_o1  (Mllama fine-tune)        -> part4/llamav_models.py    (2048 tok)
    * Qwen family (qwen2_5_vl, r1_onevision[_nothink]) -> part4/qwen_models.py (4096 tok)

Output:  mss_<model>_responses.jsonl   (JSONL, per-uid resume)

  python run_inference.py --model llava_cot --debug_n 2
  python run_inference.py --model llava_cot --data_root ~/experiments/part9/data
"""
import os
import sys
import json
import argparse

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                    # .../llava_cot_eval
sys.path.insert(0, HERE)                                         # mss_prompts
sys.path.insert(0, os.path.join(REPO, "code"))                   # run_eval
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))   # qwen_models, llamav_models

import run_eval as RE                                          # noqa: E402  (chdir's to REPO root)
from mss_prompts import build_prompt                          # noqa: E402

LLAMA_MODELS = ["llava_cot", "base_llama"]
LLAMAV_MODELS = ["llamav_o1"]
QWEN_MODELS = ["qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]
ALL_MODELS = LLAMA_MODELS + LLAMAV_MODELS + QWEN_MODELS
QWEN_MAX_NEW_TOKENS = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--subset", default="/home/ch169788/experiments/part9/mss_subset.jsonl",
                    help="manifest from prepare_subset.py")
    ap.add_argument("--data_root", default="/home/ch169788/experiments/part9/data",
                    help="dir containing chat/ and embodied/ images")
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part9/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N items and PRINT prompt+response (nothing written)")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0

    with open(args.subset) as f:
        items = [json.loads(l) for l in f]
    if debug:
        items = items[:args.debug_n]

    out_path = os.path.join(args.output_dir, "mss_%s_responses.jsonl" % args.model)
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["uid"])
                except Exception:
                    pass

    fam = "Qwen/advisor(4096)" if args.model in QWEN_MODELS else "Llama/run_eval(2048)"
    print("=" * 80, flush=True)
    print("  Part9 MSSBench INFER | model=%s [%s] | %d items%s  [NO JUDGE]"
          % (args.model, fam, len(items), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

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
    for it in items:
        if it["uid"] in written:
            continue
        prompt = build_prompt(it["task"], it["text"])
        image = Image.open(os.path.join(args.data_root, it["image"])).convert("RGB")

        resp = generate(image, prompt)

        rec = dict(it)                       # carry all manifest fields through
        rec["model"] = args.model
        rec["prompt"] = prompt               # exact model input text (prompt + query/instruction)
        rec["response"] = resp
        n_done += 1

        if debug:
            print("\n----- uid=%s [%s/%s/%s] -----" % (it["uid"], it["task"], it["category"], it["context"]))
            print("PROMPT:", prompt[:400])
            print("RESPONSE:\n", resp)
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 10 == 0:
                print("  %d generated" % n_done, flush=True)

    if debug:
        print("\n[DEBUG] %d responses printed above — nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
