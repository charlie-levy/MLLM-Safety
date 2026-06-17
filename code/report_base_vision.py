#!/usr/bin/env python
"""
report_base_vision.py — Print final table for the base Vision-Instruct eval.

Reads results/base_vision_eval/<cond>/metrics.json for each blur level and
prints one table: blur_pct × {ASR, ORR XSTest, ORR MMSA, ORR Avg, SQA}.

Usage:
  python code/report_base_vision.py
  python code/report_base_vision.py results_newton/base_vision_eval
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUR_PCTS = [0, 20, 40, 60, 80, 100]
SEVERITY_LABEL = {0: "sev0 (clean)", 20: "sev1", 40: "sev2",
                  60: "sev3", 80: "sev4", 100: "sev5"}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "base_vision_eval")

    rows = []
    for pct in BLUR_PCTS:
        cond = "clean" if pct == 0 else ("blur%d" % pct)
        path = os.path.join(root, cond, "metrics.json")
        if not os.path.exists(path):
            rows.append((pct, None))
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append((pct, m))

    print()
    print("=" * 90)
    print("  Llama-3.2-11B-Vision-Instruct (no safety)  |  String-matching metrics")
    print("=" * 90)
    print("  %-16s %10s %10s %10s %10s %10s" % (
        "Condition", "ASR", "ORR XSTest", "ORR MMSA", "ORR Avg", "SQA Acc"))
    print("  " + "-" * 68)
    for pct, m in rows:
        label = SEVERITY_LABEL[pct]
        if m is None:
            print("  %-16s  (missing)" % label)
            continue
        print("  %-16s %9.2f%% %9.2f%% %9.2f%% %9.2f%% %9.2f%%" % (
            label,
            m["asr"]["asr_pct"],
            m["xstest_orr"]["orr_pct"],
            m["mmsa_orr"]["orr_pct"],
            m["avg_orr_pct"],
            m["sqa"]["accuracy_pct"],
        ))
    print()
    print("  ASR = compliance rate (% FigStep responses with no explicit refusal)")
    print("  ORR = % benign prompts wrongly refused (XSTest) / answered No (MMSA)")
    print("  SQA = ScienceQA-250 letter-match accuracy (always clean, no blur)")
    print("  All metrics: string-matching, no LLM judge.")
    print()

    missing = [SEVERITY_LABEL[p] for p, m in rows if m is None]
    if missing:
        print("  [!] Missing conditions: %s (jobs may still be running)\n" % ", ".join(missing))

    out = os.path.join(root, "summary.json")
    summary = {str(p): m for p, m in rows if m is not None}
    os.makedirs(root, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  Wrote %s\n" % out)


if __name__ == "__main__":
    main()
