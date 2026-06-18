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
    def _pct(v):
        return ("%.2f%%" % v) if isinstance(v, (int, float)) else "PENDING"

    print("=" * 92)
    print("  %-12s  %-12s  %-12s  %-14s  %-14s" % (
        "Severity", "ASR str-mat", "ASR Guard", "ORR-avg str-mat", "ORR-avg LLaMA"))
    print("  " + "-" * 88)
    summary = {}
    for sev, cond in CONDS:
        m = _load(os.path.join(ROOT, cond, "metrics.json"))
        sm = _sm_asr(m)
        sm_orr = (m.get("avg_orr_pct") or (m.get("avg_orr") or {}).get("orr_pct")
                  if m else None)
        g = _load(os.path.join(ROOT, cond, "asr_guard.json"))
        g_asr = g.get("asr_pct") if g else None
        lo = _load(os.path.join(ROOT, cond, "judged_llama_orr.json"))
        l_orr = lo.get("avg_orr_pct") if lo else None
        print("  %-12s  %-12s  %-12s  %-14s  %-14s" % (
            "%s (%s)" % (sev, cond),
            ("%.1f%%" % sm) if sm is not None else "n/a",
            _pct(g_asr),
            ("%.1f%%" % sm_orr) if sm_orr is not None else "n/a",
            _pct(l_orr)))
        summary[sev] = {"condition": cond, "asr_string_match": sm, "asr_guard": g_asr,
                        "orr_avg_string_match": sm_orr, "orr_avg_llama": l_orr}
    print("=" * 92)
    print("  str-mat = string-match (counts deflections / 'too blurry' as success — unreliable)")
    print("  Guard   = Llama Guard 3 (true ASR) | LLaMA = LLaMA-3-8B ORR judge (true ORR)")
    print("  Run submit_base_vision_guard.sh (ASR) + submit_base_vision_orr.sh (ORR) to fill these.")
    print()
    out = os.path.join(ROOT, "asr_string_vs_guard.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    print("  wrote %s\n" % out)


if __name__ == "__main__":
    main()
