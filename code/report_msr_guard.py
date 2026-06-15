#!/usr/bin/env python
"""
report_msr_guard.py — Final table for the MSR-Align Guard eval.

Reads the aggregate JSONs written by judge_figstep_guard.py (ASR) and
judge_safety_hf.py --mode orr (ORR) for the clean and blur20 conditions and
prints a single table:

  condition × { ASR (Llama Guard), ORR-XSTest, ORR-MMSA, ORR-Avg }

Also writes results/msr_guard_eval/summary.json. CPU-only — safe to run on the
login node (with OPENBLAS_NUM_THREADS=1 to dodge the process cap), no GPU.

Usage:
  python code/report_msr_guard.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.join("results", "msr_guard_eval")
CONDITIONS = ["clean", "blur20"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(cond):
    """Return a dict of metrics for one condition, or None markers where missing."""
    asr = _load(os.path.join(ROOT, cond, "asr_guard.json"))
    orr = _load(os.path.join(ROOT, cond, "judged_llama_orr.json"))
    out = {
        "asr_pct":   asr.get("asr_pct") if asr else None,
        "asr_n":     asr.get("n_total") if asr else None,
        "xstest_orr": (orr.get("xstest", {}).get("orr_pct") if orr else None),
        "mmsa_orr":   (orr.get("mmsa", {}).get("orr_pct") if orr else None),
        "avg_orr":    (orr.get("avg_orr_pct") if orr else None),
        "asr_judge":  asr.get("judge") if asr else None,
        "orr_judge":  orr.get("judge") if orr else None,
    }
    return out


def fmt(v, suffix="%"):
    return ("%.2f%s" % (v, suffix)) if isinstance(v, (int, float)) else "  n/a "


def main():
    data = {cond: collect(cond) for cond in CONDITIONS}

    header = ("Condition", "ASR (Guard)", "ORR XSTest", "ORR MMSA", "ORR Avg")
    print()
    print("=" * 72)
    print("  MSR-Align  ·  Llama Guard 3 Vision ASR  +  LLaMA-3 ORR judge")
    print("=" * 72)
    print("  %-10s %12s %12s %12s %10s" % header)
    print("  " + "-" * 68)
    for cond in CONDITIONS:
        m = data[cond]
        print("  %-10s %12s %12s %12s %10s" % (
            cond, fmt(m["asr_pct"]), fmt(m["xstest_orr"]),
            fmt(m["mmsa_orr"]), fmt(m["avg_orr"])))
    print("=" * 72)
    print("  ASR  = %% of FigStep responses Llama Guard labels UNSAFE (lower = safer)")
    print("  ORR  = %% over-refusal on benign XSTest/MMSA  (lower = less over-refusal)")
    asr_judge = data["clean"]["asr_judge"] or data["blur20"]["asr_judge"]
    orr_judge = data["clean"]["orr_judge"] or data["blur20"]["orr_judge"]
    print("  ASR judge: %s" % (asr_judge or "n/a"))
    print("  ORR judge: %s" % (orr_judge or "n/a"))
    print()

    missing = [c for c in CONDITIONS
               if any(data[c][k] is None for k in ("asr_pct", "xstest_orr", "mmsa_orr", "avg_orr"))]
    if missing:
        print("  [!] incomplete results for: %s "
              "(judge jobs may still be running)\n" % ", ".join(missing))

    out_path = os.path.join(ROOT, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("  wrote %s\n" % out_path)


if __name__ == "__main__":
    main()
