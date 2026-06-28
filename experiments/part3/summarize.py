#!/usr/bin/env python
"""
summarize.py — Part 3 INVENTORY: how many responses were generated for each
{MM-SafetyBench-Tiny, VLS-Bench, HoliSafe} x corruption cell (LLaVA-CoT+TIS).

These are responses ONLY (no judge) — ASR/safety scoring is done separately by
your own judge program. This just confirms coverage and flags perception failures
(string-match heuristic, not a judge).

  python experiments/part3/summarize.py
  python experiments/part3/summarize.py --results_dir /home/ch169788/experiments/part3/results
"""
import os
import json
import argparse

CORRS = ["elastic_transform", "contrast", "frost", "defocus_blur", "glass_blur",
         "motion_blur", "zoom_blur", "snow", "fog", "jpeg_compression"]
DATASETS = ["mmsafety_tiny", "vls_bench", "holisafe"]


def load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/home/ch169788/experiments/part3/results")
    args = ap.parse_args()
    R = args.results_dir

    print("=" * 78)
    print("PART 3 responses generated (LLaVA-CoT+TIS) — n responses [perception-fail]")
    print("=" * 78)
    print("%-20s %-18s %-18s %-18s" % ("corruption", "mmsafety_tiny", "vls_bench", "holisafe"))
    grand = 0
    for c in CORRS:
        cells = []
        for ds in DATASETS:
            recs = load(os.path.join(R, "%s_%s_tis_responses.jsonl" % (ds, c)))
            if recs is None:
                cells.append("%-18s" % "MISSING")
            else:
                pf = sum(1 for r in recs if r.get("perception_failure"))
                grand += len(recs)
                cells.append("%-18s" % ("%d  [%d]" % (len(recs), pf)))
        print("%-20s %s %s %s" % (c, cells[0], cells[1], cells[2]))
    print("\n%d total responses across all cells. Judge these with your own program." % grand)


if __name__ == "__main__":
    main()
