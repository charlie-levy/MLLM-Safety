#!/usr/bin/env python
"""
report_grid_guards.py — Aggregate the per-cell LLaMA-Guard + WildGuard outputs
from judge_grid_guards.py into one apples-to-apples ASR table.

Reads  <out_dir>/llamaguard/*.summary.json  and  <out_dir>/wildguard/*.summary.json
(plus the per-item *.json to compute judge agreement), and writes/prints:

  * a 12-cell table: model x dataset x condition -> ASR(LLaMA-Guard), ASR(WildGuard),
    their difference, and per-item agreement (% + Cohen's kappa).
  * <out_dir>/asr_summary.json  — machine-readable, everything above.

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


def agreement(a, b):
    """Return (% agreement, cohen_kappa or None) over aligned flag lists."""
    if not a or not b or len(a) != len(b):
        return None, None
    same = sum(1 for x, y in zip(a, b) if x == y)
    pct = round(100.0 * same / len(a), 2)
    kappa = None
    if cohen_kappa_score is not None and len(set(a)) > 0:
        try:
            kappa = round(float(cohen_kappa_score(a, b)), 3)
        except Exception:
            kappa = None
    return pct, kappa


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
    for stem in stems:
        s_lg, s_wg = lg.get(stem), wg.get(stem)
        ref = s_lg or s_wg
        lg_flags = load_flags(lg_dir, stem)
        wg_flags = load_flags(wg_dir, stem)
        agr_pct, kappa = agreement(lg_flags, wg_flags)
        rows.append({
            "stem": stem,
            "model": ref.get("model", ""),
            "dataset": ref.get("dataset", ""),
            "condition": ref.get("condition", ""),
            "n_total": ref.get("n_total", 0),
            "asr_llamaguard": s_lg.get("asr_pct") if s_lg else None,
            "asr_wildguard": s_wg.get("asr_pct") if s_wg else None,
            "n_unsafe_llamaguard": s_lg.get("n_flag") if s_lg else None,
            "n_harmful_wildguard": s_wg.get("n_flag") if s_wg else None,
            "agreement_pct": agr_pct,
            "cohen_kappa": kappa,
            "parse_err_llamaguard": s_lg.get("n_parse_error") if s_lg else None,
            "parse_err_wildguard": s_wg.get("n_parse_error") if s_wg else None,
        })

    def fmt(v, suf=""):
        return "%6.2f%s" % (v, suf) if isinstance(v, (int, float)) else "   -  "

    print("\n  ASR (text-only judges) — % of responses flagged harmful\n")
    hdr = "  %-11s %-12s %-7s %5s | %8s %8s %7s | %6s %6s" % (
        "model", "dataset", "cond", "N", "LlmGuard", "WildGd", "Δ", "agree%", "kappa")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(rows, key=lambda x: (x["model"], x["dataset"], x["condition"])):
        diff = (r["asr_llamaguard"] - r["asr_wildguard"]) \
            if (r["asr_llamaguard"] is not None and r["asr_wildguard"] is not None) else None
        print("  %-11s %-12s %-7s %5s | %s %s %s | %s %s" % (
            r["model"], r["dataset"], r["condition"], r["n_total"],
            fmt(r["asr_llamaguard"]), fmt(r["asr_wildguard"]), fmt(diff),
            fmt(r["agreement_pct"]),
            ("%6.3f" % r["cohen_kappa"]) if r["cohen_kappa"] is not None else "   -  "))

    # Pooled agreement across the whole grid.
    all_lg, all_wg = [], []
    for stem in stems:
        a, b = load_flags(lg_dir, stem), load_flags(wg_dir, stem)
        if a and b and len(a) == len(b):
            all_lg += a
            all_wg += b
    pooled_pct, pooled_kappa = agreement(all_lg, all_wg)
    if pooled_pct is not None:
        print("\n  Pooled judge agreement: %.2f%%  (Cohen's kappa = %s, N=%d)" % (
            pooled_pct, pooled_kappa, len(all_lg)))

    summary = {"out_dir": out_dir, "cells": rows,
               "pooled_agreement_pct": pooled_pct, "pooled_cohen_kappa": pooled_kappa,
               "pooled_n": len(all_lg)}
    sp = os.path.join(out_dir, "asr_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n  wrote %s\n" % sp)


if __name__ == "__main__":
    main()
