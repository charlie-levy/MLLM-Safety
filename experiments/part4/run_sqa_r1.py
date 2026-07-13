#!/usr/bin/env python
"""
run_sqa_r1.py — ScienceQA UTILITY for R1-Onevision, reasoning ON vs OFF.

Answers "what does turning thinking off cost R1-Onevision on a utility benchmark?".
Runs the SAME ScienceQA-250 set the prompt-defense uses, through the SAME tested Qwen
path as Part 4 (qwen_models.load_qwen / generate_one_qwen, greedy, 4096 tok). Reasoning
is suppressed for r1_onevision_nothink exactly as in run_inference.py (no_think prefill).

Emits raw_<model>.jsonl in the EXACT schema extract_sqa.py + score_sqa.py consume:
    idx, question, choices (list), answer_idx (int), has_image (bool), response
so scoring is the ScienceQA paper-exact method (get_pred_idx), NOT an LLM judge. The
prompt->(question, choices, answer_idx) parser is verified against all 250 items.

  python run_sqa_r1.py --model r1_onevision
  python run_sqa_r1.py --model r1_onevision_nothink --debug_n 2
Then (scoring; extract needs GPU, score is CPU):
  python ../part7_module_ablation/eval/extract_sqa.py --dir <output_dir>
  python ../part7_module_ablation/eval/score_sqa.py  --dir <output_dir> --glob 'extracted_*.jsonl'
"""
import os
import sys
import re
import json
import argparse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, HERE)                                 # qwen_models (same dir)

from qwen_models import load_qwen, generate_one_qwen     # noqa: E402  (advisor's tested Qwen path)

SQA_JSON = os.path.join(REPO, "datasets", "scienceqa_250.json")
OPTIONS = ["A", "B", "C", "D", "E"]
QWEN_MAX_NEW_TOKENS = 4096                               # same as Part 4 Qwen path


def parse_sqa_item(prompt, label):
    """'Question: ..\\nChoices: (A) x (B) y ..\\nAnswer:' + letter -> (question, [choices], answer_idx).
    Verified failure-free on all 250 ScienceQA-250 items (letters sequential, answer_idx in range)."""
    m = re.search(r'Question:\s*(.*?)\s*\nChoices:', prompt, re.DOTALL)
    question = m.group(1).strip() if m else prompt.strip()
    cm = re.search(r'Choices:\s*(.*?)(?:\s*\nAnswer:|\s*$)', prompt, re.DOTALL)
    seg = cm.group(1) if cm else ""
    parts = re.split(r'\(([A-E])\)\s*', seg)
    choices = [parts[i + 1].strip() for i in range(1, len(parts), 2)]
    answer_idx = OPTIONS.index(label) if label in OPTIONS else -1
    return question, choices, answer_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["r1_onevision", "r1_onevision_nothink"])
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part_r1_sqa")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT responses (nothing written)")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    no_think = args.model.endswith("_nothink")
    out_path = os.path.join(args.output_dir, "raw_%s.jsonl" % args.model)
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    d = json.load(open(SQA_JSON))
    items = [d[k] for k in sorted(d, key=lambda x: int(x))]
    if debug:
        items = items[:args.debug_n]

    # per-idx resume
    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    print("=" * 80, flush=True)
    print("  R1-SQA utility | model=%s (no_think=%s) | %d items%s"
          % (args.model, no_think, len(items), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    model, processor = load_qwen(args.model)

    n_done = 0
    for i, it in enumerate(items):
        idx = it.get("idx", i)
        if idx in written:
            continue
        question, choices, answer_idx = parse_sqa_item(it["prompt"], it["label"])
        ip = it.get("image_path")
        has_image = bool(ip)
        image = Image.open(ip).convert("RGB") if has_image and os.path.exists(ip) else None

        resp = generate_one_qwen(model, processor, image, it["prompt"],
                                 max_new_tokens=QWEN_MAX_NEW_TOKENS, no_think=no_think)

        rec = {
            "idx": idx,
            "model": args.model,
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "has_image": has_image,
            "response": resp,
        }
        n_done += 1

        if debug:
            gold = choices[answer_idx] if 0 <= answer_idx < len(choices) else "?"
            print("\n-- idx=%s answer_idx=%d gold=%r" % (idx, answer_idx, gold))
            print("RESP:", resp[:400])
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 25 == 0:
                print("  %d generated" % n_done, flush=True)

    if debug:
        print("\n[DEBUG] %d responses printed above — nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
