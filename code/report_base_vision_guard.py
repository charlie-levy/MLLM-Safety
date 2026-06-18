#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_base_vision_guard.py — base Llama-3.2-11B-Vision ASR: string-match vs Llama Guard.

Shows, per severity 0-5, the OLD string-match ASR (from metrics.json, which counts
"list not provided / too blurry" deflections as attack successes) next to the TRUE
Llama Guard 3 ASR (from asr_guard.json produced by submit_base_vision_guard.sh).

Run after the Guard re-scoring jobs finish. CPU-only.
  python code/report_base_vision_guard.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.join("results", "base_vision_eval")
CONDS = [("sev0", "clean"), ("sev1", "blur20"), ("sev2", "blur40"),
         ("sev3", "blur60"), ("sev4", "blur80"), ("sev5", "blur100")]


def _load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _sm_asr(m):
    if not m:
        return None
    a = m.get("asr")
    return a.get("asr_pct") if isinstance(a, dict) else a


def main():
    print()
    print("=" * 78)
    print("  Base Llama-3.2-11B-Vision FigStep ASR — string-match vs Llama Guard 3")
    print("=" * 78)
    print("  %-12s  %-18s  %-18s  %s" % (
        "Severity", "ASR string-match", "ASR Llama Guard", "Guard unsafe/total"))
    print("  " + "-" * 74)
    summary = {}
    for sev, cond in CONDS:
        sm = _sm_asr(_load(os.path.join(ROOT, cond, "metrics.json")))
        g = _load(os.path.join(ROOT, cond, "asr_guard.json"))
        g_asr = g.get("asr_pct") if g else None
        g_cnt = ("%d/%d" % (g["n_unsafe"], g["n_total"])) if g else "—"
        print("  %-12s  %-18s  %-18s  %s" % (
            "%s (%s)" % (sev, cond),
            ("%.1f%%" % sm) if sm is not None else "n/a",
            ("%.2f%%" % g_asr) if g_asr is not None else "PENDING (run job)",
            g_cnt))
        summary[sev] = {"condition": cond, "asr_string_match": sm, "asr_guard": g_asr}
    print("=" * 78)
    print("  string-match = % responses with no refusal phrase (counts deflections as success)")
    print("  Llama Guard  = % responses Guard labels UNSAFE (true harmful-content rate)")
    print()
    out = os.path.join(ROOT, "asr_string_vs_guard.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    print("  wrote %s\n" % out)


if __name__ == "__main__":
    main()
