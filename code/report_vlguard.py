#!/usr/bin/env python
"""
report_vlguard.py — Final table for the VLGuard (LLaVA-1.5) Guard eval.

Reads the aggregate JSONs written by judge_figstep_guard.py (ASR) and
judge_safety_hf.py --mode orr (ORR) for each variant × condition and prints one
table:

  variant · condition × { ASR (Llama Guard), ORR-XSTest, ORR-MMSA, ORR-Avg }

Also writes results/vlguard_eval/summary.json. CPU-only — safe on the login node
(with OPENBLAS_NUM_THREADS=1 to dodge the process cap), no GPU.

Usage:
  python code/report_vlguard.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VLGUARD_VARIANTS   # noqa: E402

ROOT = os.path.join("results", "vlguard_eval")
VARIANTS = sorted(VLGUARD_VARIANTS)
CONDITIONS = ["clean", "blur20", "blur40"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(variant, cond):
    asr = _load(os.path.join(ROOT, variant, cond, "asr_guard.json"))
    orr = _load(os.path.join(ROOT, variant, cond, "judged_llama_orr.json"))
    return {
        "asr_pct":    asr.get("asr_pct") if asr else None,
        "asr_n":      asr.get("n_total") if asr else None,
        "xstest_orr": (orr.get("xstest", {}).get("orr_pct") if orr else None),
        "mmsa_orr":   (orr.get("mmsa", {}).get("orr_pct") if orr else None),
        "avg_orr":    (orr.get("avg_orr_pct") if orr else None),
        "asr_judge":  asr.get("judge") if asr else None,
        "orr_judge":  orr.get("judge") if orr else None,
    }


def fmt(v, suffix="%"):
    return ("%.2f%s" % (v, suffix)) if isinstance(v, (int, float)) else "  n/a "


def main():
    data = {v: {c: collect(v, c) for c in CONDITIONS} for v in VARIANTS}

    header = ("Variant", "Condition", "ASR (Guard)", "ORR XSTest", "ORR MMSA", "ORR Avg")
    print()
    print("=" * 82)
    print("  VLGuard (LLaVA-1.5-7B)  ·  Llama Guard 3 Vision ASR  +  LLaMA-3 ORR judge")
    print("=" * 82)
    print("  %-9s %-10s %12s %12s %12s %10s" % header)
    print("  " + "-" * 78)
    for v in VARIANTS:
        for c in CONDITIONS:
            m = data[v][c]
            print("  %-9s %-10s %12s %12s %12s %10s" % (
                v, c, fmt(m["asr_pct"]), fmt(m["xstest_orr"]),
                fmt(m["mmsa_orr"]), fmt(m["avg_orr"])))
        print("  " + "-" * 78)
    print("  ASR  = %% of FigStep responses Llama Guard labels UNSAFE (lower = safer)")
    print("  ORR  = %% over-refusal on benign XSTest/MMSA  (lower = less over-refusal)")
    any_asr = next((data[v][c]["asr_judge"] for v in VARIANTS for c in CONDITIONS
                    if data[v][c]["asr_judge"]), None)
    any_orr = next((data[v][c]["orr_judge"] for v in VARIANTS for c in CONDITIONS
                    if data[v][c]["orr_judge"]), None)
    print("  ASR judge: %s" % (any_asr or "n/a"))
    print("  ORR judge: %s" % (any_orr or "n/a"))
    print()

    missing = ["%s/%s" % (v, c) for v in VARIANTS for c in CONDITIONS
               if any(data[v][c][k] is None
                      for k in ("asr_pct", "xstest_orr", "mmsa_orr", "avg_orr"))]
    if missing:
        print("  [!] incomplete results for: %s "
              "(some jobs may still be running)\n" % ", ".join(missing))

    out_path = os.path.join(ROOT, "summary.json")
    os.makedirs(ROOT, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("  wrote %s\n" % out_path)


if __name__ == "__main__":
    main()
