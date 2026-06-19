#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_asr_beavertails.py — PHASE 5: print the full BeaverTails-V ASR grid.

Reads all available cells and recomputes ASR with the SAME scorer (metrics.is_refusal)
for every cell, so old and new are scored identically:

  * 2 existing cells (base/clean, base/blur20, base+TIS/clean) -> recomputed from the
    old responses_*.csv (full_response column). Cross-checked against the published
    76.95 / 77.80 / 21.36 -> hard error if they don't reproduce.
  * 7 new cells -> the run_beavertails.py JSONs (model_response field).

Every source is asserted to have exactly 1180 entries. CPU-only.

  python gap_beaver_6_19/calc_asr_beavertails.py
"""
import os
import sys
import csv
import json

if sys.version_info[0] < 3:
    sys.exit("ERROR: run with Python 3 — the login node's bare `python` is Python 2. "
             "Do `conda activate REU` first (or use `python3`).")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)
csv.field_size_limit(10 ** 7)

from metrics import is_refusal           # noqa: E402  (byte-identical to code/)

N_EXPECTED = 1180
MODELS = [("base", "LLaVA-CoT"), ("tis", "+ TIS")]
CONDS = ["clean", "noise20", "noise40", "blur20", "blur40"]

# existing cells -> old per-sample CSV (searched in these dirs) + published value
OLD_CSV = {
    ("base", "clean"):  ("responses_base_clean.csv",     76.95),
    ("base", "blur20"): ("responses_base_blur20.csv",    77.80),
    ("tis",  "clean"):  ("responses_base_tis_clean.csv", 21.36),
}
CSV_DIRS = ["results/new_attacks/beavertails",
            "results_6_18/task2_new_attacks/beavertails"]
JSON_DIRS = [os.path.join(HERE, "outputs"),
             "results/new_attacks/beavertails"]


def asr_from_csv(fname):
    for d in CSV_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            assert len(rows) == N_EXPECTED, "%s has %d rows (expected %d)" % (p, len(rows), N_EXPECTED)
            nonref = sum(0 if is_refusal(r["full_response"]) else 1 for r in rows)
            return 100.0 * nonref / N_EXPECTED, p
    return None, None


def asr_from_json(model, cond):
    for d in JSON_DIRS:
        p = os.path.join(d, "results_beavertails_%s_%s.json" % (model, cond))
        if os.path.exists(p):
            data = json.load(open(p, encoding="utf-8"))
            assert len(data) == N_EXPECTED, "%s has %d entries (expected %d)" % (p, len(data), N_EXPECTED)
            nonref = sum(0 if is_refusal(e["model_response"]) else 1 for e in data)
            return 100.0 * nonref / N_EXPECTED, p
    return None, None


def cell(model, cond):
    """A fresh JSON wins; else recompute the existing cell from its CSV; else None."""
    asr, _ = asr_from_json(model, cond)
    if asr is not None:
        return asr
    if (model, cond) in OLD_CSV:
        fname, pub = OLD_CSV[(model, cond)]
        asr, src = asr_from_csv(fname)
        if asr is not None:
            if abs(asr - pub) > 0.01:
                sys.exit("CROSS-CHECK FAILED: %s/%s recomputed %.2f != published %.2f (%s)"
                         % (model, cond, asr, pub, src))
            return asr
    return None


def main():
    grid = {(m, c): cell(m, c) for m, _ in MODELS for c in CONDS}

    w = 9
    print("\n  BeaverTails-V — ASR (STRING-MATCH, n=%d)\n" % N_EXPECTED)
    head = "  %-10s | " % "Model" + " | ".join("%-*s" % (w, c.capitalize()) for c in CONDS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for mkey, mlabel in MODELS:
        cells = []
        for c in CONDS:
            v = grid[(mkey, c)]
            cells.append("%.2f%%" % v if v is not None else "—")
        print("  %-10s | " % mlabel + " | ".join("%-*s" % (w, x) for x in cells))
    print()

    missing = [(m, c) for m, _ in MODELS for c in CONDS if grid[(m, c)] is None]
    if missing:
        print("  pending: " + ", ".join("%s/%s" % mc for mc in missing) + "\n")
    else:
        print("  GRID COMPLETE — all 10 cells scored (existing cells reproduced exactly).\n")


if __name__ == "__main__":
    main()
