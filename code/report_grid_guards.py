#!/usr/bin/env python
"""
report_grid_guards.py — Aggregate the per-cell LLaMA-Guard + WildGuard outputs
from judge_grid_guards.py into one apples-to-apples ASR table.

Reads  <out_dir>/llamaguard/*.summary.json  and  <out_dir>/wildguard/*.summary.json
(plus the per-item *.json to align the two judges position-by-position), and
writes/prints, per cell (model x dataset x condition):

  LG     = ASR by LLaMA-Guard (% flagged unsafe)
  WG     = ASR by WildGuard   (% flagged harmful)
  Both   = ASR where BOTH judges flag  (intersection — high-precision "definitive harm")
  Either = ASR where EITHER judge flags (union — upper bound)
  agree% = per-item agreement,  kappa = Cohen's kappa (blank when a cell is all-safe)

  <out_dir>/asr_summary.json  — machine-readable version of all of the above.

Usage:
  python code/report_grid_guards.py results/grid_guard_eval
"""
import os
import sys
import json
import glob

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None


def load_summaries(jdir):
    out = {}
    for p in glob.glob(os.path.join(jdir, "*.summary.json")):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        out[s["stem"]] = s
    return out


def load_flags(jdir, stem):
    p = os.path.join(jdir, "%s.json" % stem)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return [int(r["flag"]) for r in json.load(f)]


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


def combine(a, b):
    """From two aligned flag lists, return dict of asr_lg/wg/both/either/agree/kappa."""
    if not a or not b or len(a) != len(b):
        return None
    n = len(a)
    lg = sum(a)
    wg = sum(b)
    both = sum(1 for x, y in zip(a, b) if x and y)
    either = sum(1 for x, y in zip(a, b) if x or y)
    same = sum(1 for x, y in zip(a, b) if x == y)
    # Cohen's kappa is undefined when either rater used a single class (all-safe
    # cells); skip it then so we don't emit nan + sklearn warnings.
    kappa = None
    if cohen_kappa_score is not None and len(set(a)) > 1 and len(set(b)) > 1:
        try:
            kappa = round(float(cohen_kappa_score(a, b)), 3)
        except Exception:
            kappa = None
    return {
        "n_total": n,
        "asr_llamaguard": pct(lg, n), "asr_wildguard": pct(wg, n),
        "asr_both": pct(both, n), "asr_either": pct(either, n),
        "n_llamaguard": lg, "n_wildguard": wg, "n_both": both, "n_either": either,
        "agreement_pct": pct(same, n), "cohen_kappa": kappa,
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/grid_guard_eval"
    lg_dir = os.path.join(out_dir, "llamaguard")
    wg_dir = os.path.join(out_dir, "wildguard")
    lg = load_summaries(lg_dir)
    wg = load_summaries(wg_dir)
    stems = sorted(set(lg) | set(wg))
    if not stems:
        sys.exit("no *.summary.json found under %s/{llamaguard,wildguard}" % out_dir)

    rows = []
    all_lg, all_wg = [], []
    for stem in stems:
        ref = lg.get(stem) or wg.get(stem)
        a, b = load_flags(lg_dir, stem), load_flags(wg_dir, stem)
        m = combine(a, b)
        if m is None:   # one judge missing — fall back to whatever summaries exist
            m = {"n_total": ref.get("n_total", 0),
                 "asr_llamaguard": lg.get(stem, {}).get("asr_pct"),
                 "asr_wildguard": wg.get(stem, {}).get("asr_pct"),
                 "asr_both": None, "asr_either": None,
                 "agreement_pct": None, "cohen_kappa": None}
        else:
            all_lg += a
            all_wg += b
        m.update({"stem": stem, "model": ref.get("model", ""),
                  "dataset": ref.get("dataset", ""), "condition": ref.get("condition", ""),
                  "parse_err_llamaguard": lg.get(stem, {}).get("n_parse_error"),
                  "parse_err_wildguard": wg.get(stem, {}).get("n_parse_error")})
        rows.append(m)

    def f(v):
        return "%6.2f" % v if isinstance(v, (int, float)) else "   -  "

    print("\n  ASR (text-only judges) — % flagged harmful."
          "  Both = both judges agree (definitive); Either = at least one.\n")
    hdr = "  %-11s %-12s %-7s %5s | %6s %6s %6s %6s | %6s %6s" % (
        "model", "dataset", "cond", "N", "LG", "WG", "Both", "Either", "agree", "kappa")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(rows, key=lambda x: (x["model"], x["dataset"], x["condition"])):
        print("  %-11s %-12s %-7s %5s | %s %s %s %s | %s %s" % (
            r["model"], r["dataset"], r["condition"], r["n_total"],
            f(r["asr_llamaguard"]), f(r["asr_wildguard"]), f(r["asr_both"]), f(r["asr_either"]),
            f(r["agreement_pct"]),
            ("%6.3f" % r["cohen_kappa"]) if r["cohen_kappa"] is not None else "   -  "))

    pooled = combine(all_lg, all_wg) if all_lg else None
    if pooled:
        print("\n  POOLED (N=%d):  LG=%.2f%%  WG=%.2f%%  Both=%.2f%%  Either=%.2f%%"
              "  | agree=%.2f%%  kappa=%s" % (
                  pooled["n_total"], pooled["asr_llamaguard"], pooled["asr_wildguard"],
                  pooled["asr_both"], pooled["asr_either"], pooled["agreement_pct"],
                  pooled["cohen_kappa"]))

    summary = {"out_dir": out_dir, "cells": rows, "pooled": pooled}
    sp = os.path.join(out_dir, "asr_summary.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n  wrote %s\n" % sp)


if __name__ == "__main__":
    main()