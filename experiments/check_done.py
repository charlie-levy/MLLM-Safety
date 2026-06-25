#!/usr/bin/env python
"""
check_done.py — verify every Part 1 + Part 2 result file exists with the expected
row count, and flag empties / short files. Run before summarizing.

  python experiments/check_done.py
"""
import os
import json

P1 = "/home/ch169788/experiments/part1/results"
P2 = "/home/ch169788/experiments/part2/results"
CORRS = ["elastic_transform", "contrast", "frost", "defocus_blur", "glass_blur",
         "motion_blur", "zoom_blur", "snow", "fog", "jpeg_compression"]

# expected row counts (FigStep 500, SIUO 167, SQA 250; Part 2 materialized counts)
EXPECT = {}
for c in CORRS:
    EXPECT[os.path.join(P1, "figstep_%s_tis_asr.jsonl" % c)] = 500
    EXPECT[os.path.join(P1, "siuo_%s_tis_asr.jsonl" % c)] = 167
    EXPECT[os.path.join(P1, "sqa_%s_tis.jsonl" % c)] = 250
for ds, n in [("mmsafety_tiny", 168), ("spa_vl", 265), ("vls_bench", 500), ("holisafe", 494)]:
    for m in ("base", "tis"):
        EXPECT[os.path.join(P2, "%s_%s_clean.jsonl" % (ds, m))] = n


def n_rows(p):
    try:
        return sum(1 for l in open(p) if l.strip())
    except Exception:
        return None


def main():
    ok = miss = short = 0
    for path, exp in sorted(EXPECT.items()):
        n = n_rows(path)
        tag = os.path.relpath(path, "/home/ch169788/experiments")
        if n is None:
            print("MISSING  %-52s (expected %d)" % (tag, exp))
            miss += 1
        elif n < exp:
            print("SHORT    %-52s %d/%d  (resume: re-run its submit script)" % (tag, n, exp))
            short += 1
        else:
            ok += 1
    print("\n%d complete, %d short, %d missing  (of %d files)" % (ok, short, miss, len(EXPECT)))
    if not short and not miss:
        print("ALL RESULT FILES COMPLETE — ready to summarize.")


if __name__ == "__main__":
    main()
