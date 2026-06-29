#!/usr/bin/env python
"""
summarize.py — Part 4 INVENTORY: how many responses were generated for each
{model} x {condition} cell on SIUO. Responses ONLY (no judge) — safety scoring is
done separately by your own program.

  python experiments/part4/summarize.py
  python experiments/part4/summarize.py --results_dir /home/ch169788/experiments/part4/results
"""
import os
import json
import argparse

CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
MODELS = ["llava_cot", "base_llama", "r1_onevision", "qwen2_5_vl"]


def load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/home/ch169788/experiments/part4/results")
    args = ap.parse_args()
    R = args.results_dir

    print("=" * 86)
    print("PART 4 responses on SIUO (no judge) — n responses [perception-fail]")
    print("=" * 86)
    print("%-14s %-17s %-17s %-17s %-17s" % ("condition", *MODELS))
    grand = 0
    for c in CONDITIONS:
        cells = []
        for m in MODELS:
            recs = load(os.path.join(R, "siuo_%s_%s_responses.jsonl" % (c, m)))
            if recs is None:
                cells.append("%-17s" % "MISSING")
            else:
                pf = sum(1 for r in recs if r.get("perception_failure"))
                grand += len(recs)
                cells.append("%-17s" % ("%d  [%d]" % (len(recs), pf)))
        print("%-14s %s %s %s %s" % (c, cells[0], cells[1], cells[2], cells[3]))
    print("\n%d total responses across all 16 cells. Judge these with your own program." % grand)


if __name__ == "__main__":
    main()
