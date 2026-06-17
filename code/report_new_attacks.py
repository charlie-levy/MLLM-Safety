#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_new_attacks.py — Task 2 summary: ASR on SIUO / BeaverTails-V / SPA-VL
for LLaVA-CoT (base) vs base+TIS, at clean / blur20 / blur40.

Reads results/new_attacks/<dataset>/asr_<model>_<cond>.json and prints a table
comparable to the FigStep 70.4% (base) / 13.8% (TIS) numbers, then writes
results/new_attacks/new_attacks_summary.json.

CPU-only — safe on the login node.

Usage:
  python code/report_new_attacks.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["siuo", "spavl", "beavertails"]
MODELS   = [("base", "LLaVA-CoT"), ("base_tis", "base+TIS")]
CONDS    = ["clean", "blur20", "blur40"]


def _load(ds, model, cond):
    p = os.path.join("results", "new_attacks", ds, "asr_%s_%s.json" % (model, cond))
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _fmt(d):
    if not d:
        return "   n/a  "
    return "%6.2f%%" % d["asr_pct"]


def main():
    print()
    print("=" * 92)
    print("  New attack datasets — ASR (string-match, lower = safer)  |  base vs base+TIS")
    print("=" * 92)
    print("  %-14s  %-8s  %10s  %10s  %10s   %s" % (
        "Dataset", "Model", "clean", "blur20", "blur40", "n"))
    print("  " + "-" * 88)

    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for model, label in MODELS:
            cells = {c: _load(ds, model, c) for c in CONDS}
            n = next((c["n_total"] for c in cells.values() if c), None)
            print("  %-14s  %-8s  %10s  %10s  %10s   %s" % (
                ds, label,
                _fmt(cells["clean"]), _fmt(cells["blur20"]), _fmt(cells["blur40"]),
                n if n is not None else "-"))
            summary[ds][model] = {
                c: (cells[c]["asr_pct"] if cells[c] else None) for c in CONDS}
            summary[ds][model]["n_total"] = n
        print("  " + "-" * 88)

    print("  ASR = % harmful (image+question) prompts the model did NOT refuse.")
    print("  Compare to FigStep: base 70.4% / TIS 13.8% (clean).")
    print()

    out = os.path.join("results", "new_attacks", "new_attacks_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  wrote %s\n" % out)


if __name__ == "__main__":
    main()
