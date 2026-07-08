#!/usr/bin/env python
"""
score_sqa.py — ScienceQA accuracy, the PAPER'S EXACT method (no LLM judge).

Vendored verbatim from lupantech/ScienceQA (models/run_gpt3.py):
    pattern = re.compile(r'The answer is ([A-Z]).')
    res = pattern.findall(output); answer = res[0] if len(res)==1 else "FAILED"
    def get_pred_idx(prediction, choices, options):
        if prediction in options[:len(choices)]:
            return options.index(prediction)
        else:
            return random.choice(range(len(choices)))         # random fallback
    acc = correct / len(results) * 100

random is seeded (0) so the fallback is reproducible. We also report how many items
hit the fallback (extraction failed) and accuracy split by image / text-only.

CPU-only; run anywhere:
  python score_sqa.py /home/ch169788/experiments/part7/sqa/raw_llm.jsonl
  python score_sqa.py --dir /home/ch169788/experiments/part7/sqa
"""
import os
import re
import sys
import json
import glob
import random
import argparse

OPTIONS = ["A", "B", "C", "D", "E"]
PATTERN = re.compile(r'The answer is ([A-Z]).')          # paper-verbatim


def get_pred_idx(prediction, choices, options, rng):
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return rng.choice(range(len(choices)))            # paper-verbatim random fallback


def score_file(path):
    rng = random.Random(0)
    items = [json.loads(l) for l in open(path) if l.strip()]
    correct = 0
    failed = 0
    by = {"image": [0, 0], "text": [0, 0]}                # [correct, total]
    for it in items:
        if "pred_letter" in it:                          # hybrid extraction (rule|llm) from extract_sqa.py
            answer = it["pred_letter"] if it["pred_letter"] else "FAILED"
        else:
            res = PATTERN.findall(it["response"])         # paper-exact regex on raw responses
            answer = res[0] if len(res) == 1 else "FAILED"
        if answer == "FAILED":
            failed += 1
        pred_idx = get_pred_idx(answer, it["choices"], OPTIONS, rng)
        ok = (pred_idx == it["answer_idx"])
        correct += int(ok)
        k = "image" if it.get("has_image") else "text"
        by[k][1] += 1
        by[k][0] += int(ok)
    total = len(items)
    acc = 100.0 * correct / total if total else 0.0
    tag = os.path.basename(path).replace("raw_", "").replace(".jsonl", "")
    out = {
        "tag": tag,
        "metric": "ScienceQA test accuracy (paper-exact extraction)",
        "accuracy": round(acc, 2),
        "correct": correct,
        "total": total,
        "extraction_failed": failed,
        "acc_image": round(100.0 * by["image"][0] / by["image"][1], 2) if by["image"][1] else None,
        "acc_text_only": round(100.0 * by["text"][0] / by["text"][1], 2) if by["text"][1] else None,
        "n_image": by["image"][1],
        "n_text_only": by["text"][1],
    }
    outp = os.path.join(os.path.dirname(path), "judged_%s.json" % tag)
    json.dump(out, open(outp, "w"), indent=2)
    print("  %-8s accuracy=%5.2f%%  (%d/%d, extraction_failed=%d)  img=%s text=%s -> %s"
          % (tag, acc, correct, total, failed, out["acc_image"], out["acc_text_only"],
             os.path.basename(outp)))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir")
    ap.add_argument("--glob", default="raw_*.jsonl",
                    help="e.g. 'extracted_*.jsonl' to score the hybrid-extracted files")
    args = ap.parse_args()
    paths = list(args.files)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not paths:
        sys.exit("No input. Pass files or --dir.")
    print("=== ScienceQA accuracy (paper-exact) ===")
    for p in paths:
        score_file(p)


if __name__ == "__main__":
    main()
