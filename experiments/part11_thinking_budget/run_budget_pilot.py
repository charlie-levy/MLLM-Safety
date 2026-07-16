#!/usr/bin/env python
"""
run_budget_pilot.py — Part 11 PILOT: thinking-budget dose-response for SAFETY.

R1-Onevision on the first N SIUO samples (CLEAN images), sweeping the exact
thinking-token budget (TTBC-2, see budget_forcing.py). Responses ONLY — judged
afterwards by the existing SIUO GPT-4o R/C judge (judging/eval_hr_table1_siuo_
jsonl_R_C.py), which already splits reasoning (<think>...</think>) from the
conclusion (after </think>).

Budgets (one output file each):
    0        reasoning suppressed — EXACT part4 r1_onevision_nothink protocol
             (empty <think></think> prefill via generate_one_qwen(no_think=True);
             response saved WITHOUT think tags, identical to part4, so numbers
             are directly comparable)
    <int>    TTBC-2 exact budget (budget_forcing.generate_forced): suppress
             </think> + append "Wait" until the trace is exactly <int> tokens
    natural  no intervention — EXACT part4 r1_onevision protocol
             (generate_one_qwen, 4096 max tokens)

Everything reused UNCHANGED: qwen_models loader/runner (advisor's tested code),
dataset_loader.load_new_attack("siuo") (same samples/idx as part4 -> the R/C
judge joins on int(idx)==question_id), greedy/bf16 decode.

Sanity property: with greedy decoding, the `natural` file's first N responses
should reproduce part4's siuo_clean_r1_onevision_responses.jsonl byte-for-byte.

Output: <output_dir>/siuo_clean_r1onevision_budget<B>_responses.jsonl
JSONL append + per-idx resume per budget file.

  python run_budget_pilot.py --debug_n 2 --budgets 512          # print, no writes
  python run_budget_pilot.py --n_samples 50                      # full pilot
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, HERE)                                 # budget_forcing
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))   # qwen_models
sys.path.insert(0, os.path.join(REPO, "code"))           # run_eval, dataset_loader
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))

import run_eval as RE  # noqa: E402,F401  (chdir's to REPO root; needed by dataset_loader paths)
from dataset_loader import load_new_attack                # noqa: E402
from corruption_lib import apply_corruption, is_perception_failure  # noqa: E402
from qwen_models import load_qwen, generate_one_qwen      # noqa: E402
from budget_forcing import generate_forced                # noqa: E402

MODEL_KEY = "r1_onevision"
NATURAL_MAX_NEW_TOKENS = 4096      # part4's Qwen-path cap
ANSWER_TOKENS = 2048               # answer phase after a forced budget


def gen_for_budget(model, processor, samples, budget, out_dir, debug,
                   condition="clean", severity=0):
    """budget: 'natural' | int (0 = nothink). condition: 'clean' or a corruption
    name from corruption_lib (e.g. 'zoom_blur'); severity is applied only when
    condition != 'clean' (identical apply_corruption call as Parts 4/5/10)."""
    tag = str(budget)
    out_path = os.path.join(
        out_dir, "siuo_%s_r1onevision_budget%s_responses.jsonl" % (condition, tag))

    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    print("\n" + "-" * 78, flush=True)
    print("  budget=%s -> %s  (%d already done)"
          % (tag, os.path.basename(out_path), len(written)), flush=True)
    print("-" * 78, flush=True)

    n_done = 0
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        idx = meta.get("idx") or str(i)
        if idx in written:
            continue
        prompt, image = s["prompt"], s["image"]
        if condition != "clean":
            image = apply_corruption(image, condition, severity=severity)

        if budget == "natural":
            resp = generate_one_qwen(model, processor, image, prompt,
                                     max_new_tokens=NATURAL_MAX_NEW_TOKENS)
            think_tok, waits, rounds = None, None, None
        elif budget == 0:
            resp = generate_one_qwen(model, processor, image, prompt,
                                     max_new_tokens=ANSWER_TOKENS, no_think=True)
            think_tok, waits, rounds = 0, 0, 0
        else:
            resp, think_tok, waits, rounds = generate_forced(
                model, processor, image, prompt, budget, answer_tokens=ANSWER_TOKENS)

        rec = {
            "idx": idx,
            "model": MODEL_KEY,
            "dataset": "siuo",
            "condition": condition,
            "severity": severity if condition != "clean" else 0,
            "budget": tag,
            "think_tokens_actual": think_tok,
            "n_wait_insertions": waits,
            "n_forcing_rounds": rounds,
            "category": meta.get("category", ""),
            "prompt": prompt,
            "response": resp,
            "image_path": meta.get("image_path", ""),
            "perception_failure": is_perception_failure(resp),
        }
        n_done += 1

        if debug:
            print("\n----- idx=%s budget=%s  think_tokens=%s waits=%s rounds=%s -----"
                  % (idx, tag, think_tok, waits, rounds))
            print("PROMPT:", prompt[:200])
            print("RESPONSE (first 1200 chars):\n", resp[:1200])
            print("... RESPONSE (last 400 chars):\n", resp[-400:])
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 5 == 0:
                print("  [budget=%s] %d generated" % (tag, n_done), flush=True)

    print("  DONE budget=%s (%d new)" % (tag, n_done), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="0,512,2048,natural",
                    help="comma list of exact budgets; ints, plus 'natural'")
    ap.add_argument("--n_samples", type=int, default=50,
                    help="first N SIUO samples (deterministic order)")
    ap.add_argument("--condition", default="clean",
                    help="'clean' or a corruption name (e.g. zoom_blur) applied to every image")
    ap.add_argument("--severity", type=int, default=2,
                    help="corruption severity 1-5 (imagecorruptions); used only when --condition != clean")
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part11/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples per budget and PRINT (nothing written)")
    args = ap.parse_args()

    budgets = []
    for b in args.budgets.split(","):
        b = b.strip()
        if not b:
            continue
        budgets.append("natural" if b == "natural" else int(b))
    debug = args.debug_n and args.debug_n > 0

    samples = load_new_attack("siuo")[:args.n_samples]
    if debug:
        samples = samples[:args.debug_n]
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    cond_str = args.condition if args.condition == "clean" \
        else "%s(sev%d)" % (args.condition, args.severity)
    print("=" * 78, flush=True)
    print("  Part11 PILOT | SIUO %s | model=%s | budgets=%s | %d samples%s  [NO JUDGE]"
          % (cond_str, MODEL_KEY, budgets, len(samples), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 78, flush=True)

    model, processor = load_qwen(MODEL_KEY)      # load ONCE for every budget
    for b in budgets:
        gen_for_budget(model, processor, samples, b, args.output_dir, debug,
                       condition=args.condition, severity=args.severity)

    if debug:
        print("\n[DEBUG] responses printed above — check think-token counts hit the budget,"
              " Wait insertions appear, and an answer follows </think>. Nothing written.", flush=True)
    else:
        print("\nALL DONE budgets=%s" % budgets, flush=True)


if __name__ == "__main__":
    main()
