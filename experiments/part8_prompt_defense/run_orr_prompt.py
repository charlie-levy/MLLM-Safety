#!/usr/bin/env python
"""
run_orr_prompt.py — Part 8 GAP EXPERIMENT: what does the safety prompt cost on
BENIGN inputs?

WHY THIS EXISTS. Part 8 measured what the safety prompts buy (SIUO HR_C falls
8.4-20.4 points across three models) and what they cost in *accuracy* (ScienceQA,
MathVista: nothing). Neither of those can rule out the obvious null hypothesis for
any prompt containing "refuse if it could be harmful": that it simply refuses more,
benign or not. ScienceQA is a multiple-choice accuracy benchmark -- a model can
become uniformly more refusing and still score the same on it.

That matters for the paper's central comparison of defenses. Safety fine-tuning
(sec:defenses:tis) reaches a lower harmful rate than prompting does, and its
over-refusal tax IS reported (43.2% -> 82.3%). Prompting's is not measured at all,
so the two defenses cannot currently be ranked: a cheap defense that doubles
over-refusal is not cheap. This script measures it on the same two benign suites
and the same union protocol used everywhere else in the project, so the numbers
drop straight into that comparison.

DESIGN. Deliberately the smallest experiment that answers the question:
  * CLEAN images only. The question is the prompt's tax on benign requests, not
    its interaction with corruption; adding conditions multiplies cost without
    changing the answer. (The corrupted benign cells already exist, unprompted,
    in tab:benign-control.)
  * The same three models the SIUO prompt grid covers, and all three prompt
    variants, so every cell pairs 1:1 with a cell of tab:prompting.
  * Generation routes through the EXACT part-8 path -- same fold_prompt, same
    frozen Part-4 generate calls -- so the only difference from the SIUO grid is
    the dataset. --prompt none reproduces the undefended baseline.

SCORING. metrics.compute_orr, the union of an explicit-refusal string match and
the benchmark's own Yes/No verdict. This is the protocol behind every published
ORR number in the project, so these values are directly comparable to them.
Scored inline (no model judge, no API) and also written per item so the responses
can be re-scored later.

  python run_orr_prompt.py --model llava_cot --dataset xstest --prompt blur_safe --debug_n 3
  python run_orr_prompt.py --model qwen2_5_vl --dataset mmsa --prompt safety
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                    # .../llava_cot_eval
# insert() order is load-bearing: each insert(0) prepends, so the LAST insert wins
# lookups. part4 also has a run_inference.py; HERE must be inserted last so that
# `from run_inference import fold_prompt` resolves to part8's module, not part4's.
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))   # qwen_models
sys.path.insert(0, os.path.join(REPO, "code"))                   # run_eval, dataset_loader, metrics
sys.path.insert(0, HERE)                                         # prompts, run_inference (shadows part4's)

import run_eval as RE                                            # noqa: E402  (chdir's to REPO)
from dataset_loader import load_xstest, load_mmsa                # noqa: E402
from metrics import compute_orr                                  # noqa: E402
from prompts import SYSTEM_PROMPTS                               # noqa: E402
from run_inference import fold_prompt                            # noqa: E402  (identical injection)

LLAMA_MODELS = ["llava_cot", "base_llama"]
QWEN_MODELS = ["qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]
ALL_MODELS = LLAMA_MODELS + QWEN_MODELS
QWEN_MAX_NEW_TOKENS = 4096
EXPECTED = {"xstest": 250, "mmsa": 428}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--dataset", required=True, choices=["xstest", "mmsa"])
    ap.add_argument("--prompt", required=True, choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part8/orr")
    ap.add_argument("--debug_n", type=int, default=0)
    args = ap.parse_args()

    system = SYSTEM_PROMPTS[args.prompt]
    debug = args.debug_n and args.debug_n > 0
    tag = args.model if system is None else "%s_%s" % (args.model, args.prompt)
    out_path = os.path.join(args.output_dir, "orr_%s_%s_responses.jsonl" % (args.dataset, tag))
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

    samples = load_xstest() if args.dataset == "xstest" else load_mmsa()
    assert len(samples) == EXPECTED[args.dataset], \
        "%s must have %d items, got %d" % (args.dataset, EXPECTED[args.dataset], len(samples))
    if debug:
        samples = samples[:args.debug_n]

    fam = "Qwen(4096)" if args.model in QWEN_MODELS else "Llama/run_eval(2048)"
    print("=" * 80, flush=True)
    print("  Part8 ORR | %s model=%s prompt=%s [%s] | %d samples%s  [CLEAN images]"
          % (args.dataset, args.model, args.prompt, fam, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("  system_prompt = %s" % ("<none> (undefended baseline)" if system is None else repr(system)),
          flush=True)
    print("=" * 80, flush=True)

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

    records, n_done = [], 0
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        idx = meta.get("idx") or str(i)
        if idx in written:
            continue
        resp = generate(s["image"], fold_prompt(system, s["prompt"]))
        if not resp:
            raise RuntimeError("EMPTY response at idx=%s (%s/%s/%s) -- fail loudly"
                               % (idx, args.model, args.dataset, args.prompt))
        rec = {
            "idx": idx,
            "model": tag,
            "base_model": args.model,
            "dataset": args.dataset,
            "condition": "clean",
            "prompt_variant": args.prompt,
            "system_prompt": system,
            "prompt": s["prompt"],
            "response": resp,
            "image_path": meta.get("image_path", ""),
        }
        records.append(rec)
        n_done += 1
        if debug:
            print("\n----- idx=%s -----\nRESPONSE:\n %s" % (idx, resp[:600]))
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 25 == 0:
                print("  %d generated" % n_done, flush=True)

    # Score whatever is on disk (not just this run's slice), so a resumed job still
    # reports the full-cell ORR rather than the ORR of its own tail.
    scored = records
    if not debug and os.path.exists(out_path):
        scored = [json.loads(l) for l in open(out_path) if l.strip()]
    # model=None on purpose. That argument selects an ADAPTER-specific answer parser
    # (tis/sage/msr); none of these three are adapters, so auto-detect is correct and
    # is what the published ORR numbers for non-adapter models used. Passing
    # args.model would be a no-op today but would silently change scoring if these
    # names were ever added to metrics.ANSWER_PARSERS.
    orr = compute_orr(scored, dataset=args.dataset, model=None)

    # The U+2019 refusal bug (see CLAUDE.md) lives in this same string matcher. It is
    # believed harmless here, but a cell where most responses carry a curly apostrophe
    # is exactly where it would bite, so report the exposure rather than assume.
    n_curly = sum(1 for r in scored if "’" in r["response"])
    print("\n%s | %s | %s : ORR %.2f%%  (%d/%d over-refused)"
          % (args.dataset, tag, "clean", orr["orr_pct"], orr["n_refused"], orr["n_total"]), flush=True)
    print("  responses containing U+2019: %d/%d" % (n_curly, len(scored)), flush=True)

    if not debug:
        summ = os.path.join(args.output_dir, "orr_%s_%s_summary.json" % (args.dataset, tag))
        with open(summ, "w") as f:
            json.dump({"dataset": args.dataset, "model": args.model, "prompt_variant": args.prompt,
                       "tag": tag, "condition": "clean", "n_curly_apostrophe": n_curly, **orr}, f, indent=2)
        print("DONE -> %s\n     -> %s" % (out_path, summ), flush=True)
    else:
        print("\n[DEBUG] nothing written.", flush=True)


if __name__ == "__main__":
    main()
