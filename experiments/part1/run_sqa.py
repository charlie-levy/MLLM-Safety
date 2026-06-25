#!/usr/bin/env python
"""
run_sqa.py — Part 1 utility driver: LLaVA-CoT+TIS on ScienceQA under ONE
ImageNet-C corruption (severity=3), graded inline by the existing LLaMA-3-8B
judge (the same judge code/judge_sqa_utility_hf.py uses — "same as existing SQA
scripts").

Reuses unchanged:
  * model loading + frozen generation -> code/run_eval.py (RE.load / RE.generate_one)
  * SQA data                          -> datasets/scienceqa_250.json (prompt = Q+choices,
                                          label = letter, image_path)
  * utility judge                     -> code/judge_sqa_utility_hf.Judge
                                          (NousResearch/Meta-Llama-3-8B-Instruct,
                                           CORRECT / INCORRECT)
  * predicted letter (informational)  -> code/metrics.extract_answer_letter (A-D)

JSONL append + per-idx resume.

  python run_sqa.py --corruption fog
  python run_sqa.py --corruption glass_blur --debug_n 3
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
from judge_sqa_utility_hf import Judge, DEFAULT_MODEL          # noqa: E402
from metrics import extract_answer_letter                      # noqa: E402
from PIL import Image                                          # noqa: E402
from corruption_lib import apply_corruption, PART1_CORRUPTIONS  # noqa: E402

MODEL = "llava_cot_tis"   # Part 1 is TIS-only
SEVERITY = 3
SQA_JSON = "datasets/scienceqa_250.json"   # relative to REPO (RE import chdir'd us here)


def split_question_choices(prompt):
    """SQA prompt = 'Question: ...\\nChoices: (A) ... (B) ...\\nAnswer:'."""
    if "Choices:" in prompt:
        q, rest = prompt.split("Choices:", 1)
        choices = rest.replace("Answer:", "").strip()
        return q.replace("Question:", "").strip(), choices
    return prompt.strip(), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruption", required=True, choices=PART1_CORRUPTIONS)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part1/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT responses + grading")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    out_path = os.path.join(args.output_dir, "sqa_%s_tis.jsonl" % args.corruption)
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
    if debug:
        items = items[:args.debug_n]

    print("=" * 78, flush=True)
    print("  Part1 SQA | corruption=%s(sev%d) model=%s | %d samples%s"
          % (args.corruption, SEVERITY, MODEL, len(items), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 78, flush=True)

    model, processor = RE.load(MODEL)
    judge = Judge(DEFAULT_MODEL)
    judge.model_id = DEFAULT_MODEL

    n_done = n_correct = 0
    for i, it in enumerate(items):
        idx = it.get("idx", i)
        if idx in written:
            continue
        image = Image.open(it["image_path"]).convert("RGB")
        image = apply_corruption(image, args.corruption, severity=SEVERITY)

        resp = RE.generate_one(model, processor, image, it["prompt"])
        verdict = judge.ask(it["prompt"], it["label"], resp)   # True / False / None
        pred = extract_answer_letter(resp)
        question, choices = split_question_choices(it["prompt"])

        rec = {
            "idx": idx,
            "corruption": args.corruption,
            "question": question,
            "choices": choices,
            "ground_truth": it["label"],
            "response": resp,
            "predicted_answer": pred,
            "correct": bool(verdict) if verdict is not None else None,
        }
        n_done += 1
        n_correct += (verdict is True)

        if debug:
            print("\n----- sample %d  idx=%s -----" % (i, idx))
            print("Q:", question[:160], "| choices:", choices[:120], "| gt:", it["label"])
            print("RESPONSE:\n", resp)
            print("predicted=%s  judge_correct=%s" % (pred, rec["correct"]))
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 25 == 0:
                print("  %d graded (running acc %.1f%%)" % (n_done, 100.0 * n_correct / n_done), flush=True)

    acc = 100.0 * n_correct / n_done if n_done else 0.0
    if debug:
        print("\n[DEBUG] acc %.1f%% (%d/%d) — confirm responses + string-match above. Nothing written."
              % (acc, n_correct, n_done), flush=True)
    else:
        print("\nDONE -> %s  (%d new, acc %.1f%%)" % (out_path, n_done, acc), flush=True)


if __name__ == "__main__":
    main()