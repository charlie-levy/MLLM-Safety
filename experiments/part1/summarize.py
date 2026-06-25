#!/usr/bin/env python
"""
summarize.py — Part 1 results: ASR (FigStep/SIUO x 10 corruptions, TIS) +
ScienceQA utility (10 corruptions, TIS). Run after the jobs finish.

  python experiments/part1/summarize.py
  python experiments/part1/summarize.py --results_dir /home/ch169788/experiments/part1/results
"""
import os
import json
import argparse

CORRS = ["elastic_transform", "contrast", "frost", "defocus_blur", "glass_blur",
         "motion_blur", "zoom_blur", "snow", "fog", "jpeg_compression"]
DATASETS = ["figstep", "siuo"]


def load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/home/ch169788/experiments/part1/results")
    args = ap.parse_args()
    R = args.results_dir

    print("=" * 72)
    print("PART 1 ASR (LLaVA-CoT+TIS) — unsafe / total  [Llama-Guard-3-Vision]")
    print("=" * 72)
    print("%-20s %-16s %-16s %s" % ("corruption", "figstep", "siuo", "perc.fail f/s"))
    for c in CORRS:
        row = ["%-20s" % c]
        pf = []
        for ds in DATASETS:
            recs = load(os.path.join(R, "%s_%s_tis_asr.jsonl" % (ds, c)))
            if recs is None:
                row.append("%-16s" % "MISSING")
                pf.append("-")
            else:
                u = sum(1 for r in recs if r.get("llama_guard_label") == "unsafe")
                row.append("%-16s" % ("%d/%d (%.0f%%)" % (u, len(recs), 100.0 * u / max(1, len(recs)))))
                pf.append(str(sum(1 for r in recs if r.get("perception_failure"))))
        print("%s %s %s" % (row[0], row[1], row[2]) + "   " + "/".join(pf))

    print("\n" + "=" * 72)
    print("PART 1 ScienceQA utility (LLaVA-CoT+TIS) — correct / total  [LLaMA-3-8B]")
    print("=" * 72)
    print("%-20s %s" % ("corruption", "accuracy"))
    for c in CORRS:
        recs = load(os.path.join(R, "sqa_%s_tis.jsonl" % c))
        if recs is None:
            print("%-20s %s" % (c, "MISSING"))
            continue
        ok = sum(1 for r in recs if r.get("correct") is True)
        none = sum(1 for r in recs if r.get("correct") is None)
        print("%-20s %d/%d (%.0f%%)%s" % (c, ok, len(recs), 100.0 * ok / max(1, len(recs)),
                                          "   [%d ungraded]" % none if none else ""))


if __name__ == "__main__":
    main()
