#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_vlguard_sqa.py — print the VLGuard SQA-utility column (clean + blur).

Reads judged_vlguard_<variant>_sqa.json (accuracy) from each variant/condition dir,
searching the live results and the local mirror. CPU-only.

  python gap_vlguard_sqa_6_19/report_vlguard_sqa.py
"""
import os
import sys
import json

if sys.version_info[0] < 3:
    sys.exit("ERROR: run with Python 3 — `conda activate REU` first (or use python3).")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)

ROOTS = ["results/vlguard_eval", "results_newton/vlguard_eval"]
VARIANTS = ["mixed", "posthoc"]
CONDS = ["clean", "blur20", "blur40"]


def acc(variant, cond):
    for root in ROOTS:
        p = os.path.join(root, variant, cond, "judged_vlguard_%s_sqa.json" % variant)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            return d.get("accuracy"), d.get("correct"), d.get("total")
    return None, None, None


def main():
    w = 9
    print("\n  VLGuard SQA utility  ·  LLaVA-1.5-7B  ·  ScienceQA-250 accuracy (higher = better)\n")
    head = "  %-9s | " % "Variant" + " | ".join("%-*s" % (w, c.capitalize()) for c in CONDS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    pending = []
    for v in VARIANTS:
        cells = []
        for c in CONDS:
            a, corr, tot = acc(v, c)
            if a is None:
                cells.append("—")
                pending.append("%s/%s" % (v, c))
            else:
                cells.append("%.1f%%" % a)
        print("  %-9s | " % v + " | ".join("%-*s" % (w, x) for x in cells))
    print()
    if pending:
        print("  pending: " + ", ".join(pending) + "\n")
    else:
        print("  SQA column complete (all 6 cells).\n")


if __name__ == "__main__":
    main()
