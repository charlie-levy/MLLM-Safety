#!/usr/bin/env python
"""
run_sqa_multimodel.py — Part 14: ScienceQA utility under ImageNet-C corruption for
ALL the models in the main safety table, so the safety--capability decoupling can be
shown across models instead of one.

WHY THIS EXISTS. The decoupling result is the paper's stated central claim, but it is
currently measured on a single checkpoint (safety-tuned LLaVA-CoT, ten corruptions,
sec:results:decoupling). A reviewer's first question is whether decoupling is a
property of VLMs or of that one model. The safety half of the answer already exists
for six configurations x three corruptions (tab:siuo-main); the utility half does not.
This driver produces exactly that missing half, so every safety cell in Table 1 gets a
paired utility cell and the decoupling scatter can be redrawn with every model on it.

APPLES-TO-APPLES BY CONSTRUCTION. Nothing here is new science; it is part1/run_sqa.py's
utility protocol driven over part4/run_inference.py's model dispatch, both reused
unchanged:
  * SQA items + prompts    -> datasets/scienceqa_250.json      (same 250 as part1)
  * corruption + severity  -> corruption_lib.apply_corruption / severity_for
  * grading                -> judge_sqa_utility_hf.Judge (LLaMA-3-8B, CORRECT/INCORRECT)
                              -- a LOCAL judge, so this costs no API credits
  * generation             -> the SAME per-family path each model already uses:
                              run_eval (2048 tok) for Llama/LLaVA-CoT, the staged
                              4-turn pipeline for LlamaV-o1, qwen_models (4096 tok)
                              for the Qwen family, incl. the no-think prefill.
Output schema matches part1's jsonl exactly, so the existing decoupling analysis can
read both without special-casing.

  python run_sqa_multimodel.py --model qwen2_5_vl --corruption clean --debug_n 3
  python run_sqa_multimodel.py --model r1_onevision --corruption zoom_blur
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))   # qwen/llamav loaders

import run_eval as RE                                          # noqa: E402 (chdir's to REPO)
from judge_sqa_utility_hf import Judge, DEFAULT_MODEL          # noqa: E402
from metrics import extract_answer_letter                      # noqa: E402
from PIL import Image                                          # noqa: E402
from corruption_lib import apply_corruption, PART1_CORRUPTIONS, severity_for  # noqa: E402

# Same family split as part4/run_inference.py -- keep in sync if that file changes.
LLAMA_MODELS  = ["llava_cot", "base_llama"]
LLAMAV_MODELS = ["llamav_o1"]
QWEN_MODELS   = ["qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]
ALL_MODELS    = LLAMA_MODELS + LLAMAV_MODELS + QWEN_MODELS

QWEN_MAX_NEW_TOKENS = 4096
SQA_JSON = "datasets/scienceqa_250.json"   # relative to REPO (run_eval import chdir'd us)


def split_question_choices(prompt):
    """SQA prompt = 'Question: ...\\nChoices: (A) ... (B) ...\\nAnswer:'."""
    if "Choices:" in prompt:
        q, rest = prompt.split("Choices:", 1)
        return q.replace("Question:", "").strip(), rest.replace("Answer:", "").strip()
    return prompt.strip(), ""


def build_generate(model_key):
    """Return (generate_fn, family_label) using each model's already-validated path."""
    if model_key in LLAMA_MODELS:
        model, processor = RE.load(model_key)
        return (lambda img, p: RE.generate_one(model, processor, img, p)), "Llama/run_eval(2048)"
    if model_key in LLAMAV_MODELS:
        from llamav_models import load_llamav_o1, generate_llamav_staged
        model, processor = load_llamav_o1()
        # OFFICIAL 4-turn staged pipeline; single-turn only yields a planning preamble.
        return (lambda img, p: generate_llamav_staged(model, processor, img, p)), "LlamaV/staged"
    from qwen_models import load_qwen, generate_one_qwen
    model, processor = load_qwen(model_key)
    no_think = model_key.endswith("_nothink")
    return (lambda img, p: generate_one_qwen(model, processor, img, p,
                                             max_new_tokens=QWEN_MAX_NEW_TOKENS,
                                             no_think=no_think)), "Qwen(4096)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--corruption", required=True,
                    choices=["clean"] + list(PART1_CORRUPTIONS))
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part14/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT responses + grading (nothing written)")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    is_clean = (args.corruption == "clean")
    sev = 0 if is_clean else severity_for(args.corruption)

    out_path = os.path.join(args.output_dir, "sqa_%s_%s.jsonl" % (args.corruption, args.model))
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

    with open(SQA_JSON) as f:
        data = json.load(f)
    items = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
    assert len(items) == 250, "ScienceQA set must be 250 items, got %d" % len(items)
    if debug:
        items = items[:args.debug_n]

    generate, fam = build_generate(args.model)
    judge = Judge(DEFAULT_MODEL)
    judge.model_id = DEFAULT_MODEL

    print("=" * 80, flush=True)
    print("  Part14 SQA | corruption=%s(sev%d) model=%s [%s] | %d samples%s"
          % (args.corruption, sev, args.model, fam, len(items),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    n_done = n_correct = 0
    for i, it in enumerate(items):
        idx = it.get("idx", i)
        if idx in written:
            continue
        image = Image.open(it["image_path"]).convert("RGB")
        if not is_clean:
            image = apply_corruption(image, args.corruption, severity=sev)

        resp = generate(image, it["prompt"])
        if not resp:
            raise RuntimeError("EMPTY response at idx=%s (%s/%s) -- fail loudly, do not "
                               "write blanks" % (idx, args.model, args.corruption))
        verdict = judge.ask(it["prompt"], it["label"], resp)   # True / False / None
        question, choices = split_question_choices(it["prompt"])

        rec = {
            "idx": idx,
            "model": args.model,
            "corruption": args.corruption,
            "severity": sev,
            "question": question,
            "choices": choices,
            "ground_truth": it["label"],
            "response": resp,
            "predicted_answer": extract_answer_letter(resp),
            "correct": bool(verdict) if verdict is not None else None,
        }
        n_done += 1
        n_correct += (verdict is True)

        if debug:
            print("\n----- sample %d  idx=%s -----" % (i, idx))
            print("Q:", question[:160], "| gt:", it["label"])
            print("RESPONSE:\n", resp[:800])
            print("predicted=%s  judge_correct=%s" % (rec["predicted_answer"], rec["correct"]))
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 25 == 0:
                print("  %d graded (running acc %.1f%%)"
                      % (n_done, 100.0 * n_correct / n_done), flush=True)

    acc = 100.0 * n_correct / n_done if n_done else 0.0
    if debug:
        print("\n[DEBUG] acc %.1f%% (%d/%d) -- confirm responses + grading above. Nothing written."
              % (acc, n_correct, n_done), flush=True)
    else:
        print("\nDONE -> %s  (%d new, acc %.1f%%)" % (out_path, n_done, acc), flush=True)


if __name__ == "__main__":
    main()
