#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_asr_corruptions.py — print the BeaverTails-V ASR grid for the NEW corruptions
(motion_blur / jpeg / pixelate at 20% and 40%, base + TIS).

Scored with the SAME scorer as every other cell (metrics.is_refusal; non-refusal of a
harmful prompt = attack success). Clean is shown as the anchor column, recomputed from
the old responses CSV and cross-checked against the published 76.95 / 21.36 (hard error
if it doesn't reproduce — proves the scorer here is identical to the table's).

Every source is asserted to have exactly 1180 entries. CPU-only.

  python gap_beaver_6_19/calc_asr_corruptions.py
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
# clean anchor + the 6 new corruption conditions
CONDS = ["clean",
         "motion_blur20", "motion_blur40",
         "jpeg20", "jpeg40",
         "pixelate20", "pixelate40"]

# clean anchor -> old per-sample CSV (searched in these dirs) + published value
OLD_CSV = {
    ("base", "clean"): ("responses_base_clean.csv",     76.95),
    ("tis",  "clean"): ("responses_base_tis_clean.csv", 21.36),
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
    """Fresh JSON wins; else (clean only) recompute from the old CSV with a cross-check."""
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

    w = 13
    print("\n  BeaverTails-V — ASR (STRING-MATCH, n=%d) — NEW CORRUPTIONS\n" % N_EXPECTED)
    head = "  %-10s | " % "Model" + " | ".join("%-*s" % (w, c) for c in CONDS)
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
        print("  GRID COMPLETE — all 14 cells scored (clean anchors reproduced exactly).\n")


if __name__ == "__main__":
    main()
