#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_msr_guard.py — Final table + aggregates for the MSR-Align Guard eval.

For each condition (clean, blur20) it merges the ASR aggregate (asr_guard.json,
from judge_figstep_guard.py) and the ORR aggregate (judged_llama_orr.json, from
judge_safety_hf.py --mode orr) into a single per-condition aggregate, writes it,
then prints one table:

  condition × { ASR (Llama Guard), ORR-XSTest, ORR-MMSA, ORR-Avg }

Writes:
  results/msr_guard_eval/<cond>/aggregate.json   merged ASR+ORR per condition
  results/msr_guard_eval/summary.json            cross-condition summary

CPU-only (json only) — safe on the login node, no GPU, no pandas.

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

# SQA utility (LLaMA-judged) lives in its own dirs from the corruption sweeps;
# pull the MSR clean + 20%-blur numbers in so all three metrics sit in one report.
SQA_FILES = {
    "clean":  os.path.join("results", "sqa_noise_sweep", "judged_base_msr_clean.json"),
    "blur20": os.path.join("results", "sqa_blur_pct", "judged_base_msr_gaussian_blur_pct_p20.json"),
}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(cond):
    """Merge ASR + ORR aggregates for one condition; also persist aggregate.json."""
    asr = _load(os.path.join(ROOT, cond, "asr_guard.json"))
    orr = _load(os.path.join(ROOT, cond, "judged_llama_orr.json"))
    sqa = _load(SQA_FILES.get(cond, ""))

    aggregate = {
        "condition": cond,
        "asr": asr,                       # full ASR aggregate (or None if missing)
        "orr": orr,                       # full ORR aggregate (or None if missing)
        "sqa": sqa,                       # full SQA aggregate (or None if missing)
        "asr_judge": asr.get("judge") if asr else None,
        "orr_judge": orr.get("judge") if orr else None,
        "sqa_judge": sqa.get("judge") if sqa else None,
    }
    if asr is not None or orr is not None or sqa is not None:
        with open(os.path.join(ROOT, cond, "aggregate.json"), "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)

    # Flat view for the table.
    return {
        "asr_pct":    asr.get("asr_pct") if asr else None,
        "xstest_orr": (orr.get("xstest", {}).get("orr_pct") if orr else None),
        "mmsa_orr":   (orr.get("mmsa", {}).get("orr_pct") if orr else None),
        "avg_orr":    (orr.get("avg_orr_pct") if orr else None),
        "sqa_acc":    sqa.get("accuracy") if sqa else None,
        "asr_judge":  aggregate["asr_judge"],
        "orr_judge":  aggregate["orr_judge"],
        "sqa_judge":  aggregate["sqa_judge"],
    }


def fmt(v):
    return ("%.2f%%" % v) if isinstance(v, (int, float)) else "  n/a "


def main():
    data = {cond: collect(cond) for cond in CONDITIONS}

    header = ("Condition", "ASR (Guard)", "ORR XSTest", "ORR MMSA", "ORR Avg", "SQA util")
    print()
    print("=" * 84)
    print("  MSR-Align  ·  Llama Guard 3 Vision ASR  +  LLaMA-3 ORR judge  +  LLaMA SQA")
    print("=" * 84)
    print("  %-10s %12s %12s %12s %10s %10s" % header)
    print("  " + "-" * 80)
    for cond in CONDITIONS:
        m = data[cond]
        print("  %-10s %12s %12s %12s %10s %10s" % (
            cond, fmt(m["asr_pct"]), fmt(m["xstest_orr"]),
            fmt(m["mmsa_orr"]), fmt(m["avg_orr"]), fmt(m["sqa_acc"])))
    print("=" * 84)
    print("  ASR = % FigStep responses Guard labels UNSAFE (lower=safer) | "
          "ORR = % over-refusal on benign XSTest/MMSA (lower=better)")
    print("  SQA = % ScienceQA accuracy (higher=better, LLaMA-judged)")
    print("  ASR judge: %s" % (data["clean"]["asr_judge"] or data["blur20"]["asr_judge"] or "n/a"))
    print("  ORR judge: %s" % (data["clean"]["orr_judge"] or data["blur20"]["orr_judge"] or "n/a"))
    print("  SQA judge: %s" % (data["clean"]["sqa_judge"] or data["blur20"]["sqa_judge"] or "n/a"))
    print()

    missing = [c for c in CONDITIONS
               if any(data[c][k] is None for k in ("asr_pct", "xstest_orr", "mmsa_orr", "avg_orr"))]
    if missing:
        print("  [!] incomplete results for: %s "
              "(judge jobs may still be running)\n" % ", ".join(missing))

    with open(os.path.join(ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("  wrote %s\n" % os.path.join(ROOT, "summary.json"))


if __name__ == "__main__":
    main()
