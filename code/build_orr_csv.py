#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_orr_csv.py — assemble responses_orr.csv from responses_xstest.json +
responses_mmsa.json in a condition dir, so the LLaMA-3 ORR judge can score it.

Used to re-judge the Base Llama-3.2-Vision ORR with the SAME judge as MSR/VLGuard
(its string-match ORR is inconsistent: XSTest counts "I cannot…" refusals, MMSA
counts only "No" answers, so they diverge under blur).

Usage:
  python code/build_orr_csv.py results/base_vision_eval/clean
"""
import os
import sys
import csv
import json
import argparse

ORR_CSV_FIELDS = ["idx", "dataset", "category", "prompt", "image_path", "full_response"]


def load_keyed(path):
    with open(path, encoding="utf-8") as f:
        return list(json.load(f).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond_dir", help="e.g. results/base_vision_eval/clean")
    args = ap.parse_args()

    rows = []
    for short in ("xstest", "mmsa"):
        p = os.path.join(args.cond_dir, "responses_%s.json" % short)
        if not os.path.exists(p):
            raise FileNotFoundError("missing %s" % p)
        for r in load_keyed(p):
            rows.append({
                "idx":          r.get("idx", ""),
                "dataset":      r.get("dataset", short.upper()),
                "category":     r.get("category", "") or "",
                "prompt":       r.get("prompt", ""),
                "image_path":   r.get("image_path", ""),
                "full_response": r.get("full_response", ""),
            })

    out = os.path.join(args.cond_dir, "responses_orr.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ORR_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows: XSTest+MMSA)" % (out, len(rows)))


if __name__ == "__main__":
    main()
