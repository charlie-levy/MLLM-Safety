#!/usr/bin/env python
"""
summarize.py — Part 2 results: clean-image ASR for LLaVA-CoT Base vs Base+TIS
across the 4 datasets, with per-category breakdown. Run after the jobs finish.

  python experiments/part2/summarize.py
  python experiments/part2/summarize.py --results_dir /home/ch169788/experiments/part2/results
"""
import os
import json
import argparse
from collections import defaultdict

MODELS = ["base", "tis"]
DATASETS = ["mmsafety_tiny", "spa_vl", "vls_bench", "holisafe"]


def load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/home/ch169788/experiments/part2/results")
    ap.add_argument("--by_category", action="store_true", help="also print per-category ASR")
    args = ap.parse_args()
    R = args.results_dir

    print("=" * 72)
    print("PART 2 ASR (clean images) — unsafe / total  [Llama-Guard-3-Vision]")
    print("=" * 72)
    print("%-16s %-18s %-18s %s" % ("dataset", "base", "tis", "perc.fail b/t"))
    data = {}
    for ds in DATASETS:
        cells, pf = [], []
        for m in MODELS:
            recs = load(os.path.join(R, "%s_%s_clean.jsonl" % (ds, m)))
            data[(ds, m)] = recs
            if recs is None:
                cells.append("%-18s" % "MISSING")
                pf.append("-")
            else:
                u = sum(1 for r in recs if r.get("llama_guard_label") == "unsafe")
                cells.append("%-18s" % ("%d/%d (%.0f%%)" % (u, len(recs), 100.0 * u / max(1, len(recs)))))
                pf.append(str(sum(1 for r in recs if r.get("perception_failure"))))
        print("%-16s %s %s   %s" % (ds, cells[0], cells[1], "/".join(pf)))

    if args.by_category:
        print("\n" + "=" * 72)
        print("By category (unsafe / total)")
        print("=" * 72)
        for ds in DATASETS:
            print("\n[%s]" % ds)
            for m in MODELS:
                recs = data.get((ds, m))
                if not recs:
                    continue
                tot, uns = defaultdict(int), defaultdict(int)
                for r in recs:
                    c = r.get("category", "")
                    tot[c] += 1
                    uns[c] += (r.get("llama_guard_label") == "unsafe")
                print("  %s:" % m)
                for c in sorted(tot):
                    print("    %-34s %d/%d (%.0f%%)" % (c, uns[c], tot[c], 100.0 * uns[c] / tot[c]))


if __name__ == "__main__":
    main()
