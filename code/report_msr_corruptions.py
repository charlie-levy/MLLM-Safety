#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_msr_corruptions.py — Task 1 summary table for MSR-Align under the new
corruption types (JPEG compression + motion blur), alongside Gaussian blur.

Reads, per condition:
  ASR (Llama Guard)  results/msr_guard_eval/<cond>/asr_guard.json        -> asr_pct
  ORR (LLaMA-3)      results/msr_guard_eval/<cond>/judged_llama_orr.json  -> xstest/mmsa/avg
  SQA (LLaMA-3)      results/sqa_<corr>_pct/judged_base_msr_<corr>_pct_p<N>.json -> accuracy

Prints a table and writes results/msr_corruptions_summary.json.
CPU-only — safe on the login node.

Usage:
  python code/report_msr_corruptions.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (condition label, SQA judged-file path).  clean/blur are shown for context if present;
# jpeg/motion_blur are the new Task-1 conditions.
CONDITIONS = [
    ("clean",         "results/sqa_noise_sweep/judged_base_msr_clean.json"),
    ("blur20",        "results/sqa_blur_pct/judged_base_msr_gaussian_blur_pct_p20.json"),
    ("blur40",        "results/sqa_blur_pct/judged_base_msr_gaussian_blur_pct_p40.json"),
    ("jpeg20",        "results/sqa_jpeg_pct/judged_base_msr_jpeg_pct_p20.json"),
    ("jpeg40",        "results/sqa_jpeg_pct/judged_base_msr_jpeg_pct_p40.json"),
    ("motion_blur20", "results/sqa_motion_blur_pct/judged_base_msr_motion_blur_pct_p20.json"),
    ("motion_blur40", "results/sqa_motion_blur_pct/judged_base_msr_motion_blur_pct_p40.json"),
]


def _load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v):
    return ("%6.2f%%" % v) if isinstance(v, (int, float)) else "   n/a "


def main():
    print()
    print("=" * 104)
    print("  MSR-Align under image corruption  |  ASR (Llama Guard 3)  +  ORR (LLaMA-3)  +  SQA (LLaMA-3)")
    print("=" * 104)
    print("  %-14s  %10s  %10s  %10s  %10s  %10s   %s" % (
        "Condition", "ASR[sfer<]", "ORR-XS[<]", "ORR-MM[<]", "ORR-avg[<]", "SQA[>]", "status"))
    print("  " + "-" * 100)

    summary = {}
    for cond, sqa_path in CONDITIONS:
        asr = _load(os.path.join("results", "msr_guard_eval", cond, "asr_guard.json"))
        orr = _load(os.path.join("results", "msr_guard_eval", cond, "judged_llama_orr.json"))
        sqa = _load(sqa_path)

        asr_pct = asr.get("asr_pct") if asr else None
        xs_orr  = orr.get("xstest", {}).get("orr_pct") if orr else None
        mm_orr  = orr.get("mmsa", {}).get("orr_pct") if orr else None
        avg_orr = orr.get("avg_orr_pct") if orr else None
        sqa_acc = sqa.get("accuracy") if sqa else None

        have = [bool(asr), bool(orr), bool(sqa)]
        status = "complete" if all(have) else (
            "MISSING(" + ",".join(n for n, h in zip(("asr", "orr", "sqa"), have) if not h) + ")")

        print("  %-14s  %10s  %10s  %10s  %10s  %10s   %s" % (
            cond, _fmt(asr_pct), _fmt(xs_orr), _fmt(mm_orr), _fmt(avg_orr), _fmt(sqa_acc), status))

        summary[cond] = {
            "asr_pct": asr_pct,
            "orr_xstest_pct": xs_orr,
            "orr_mmsa_pct": mm_orr,
            "orr_avg_pct": avg_orr,
            "sqa_accuracy": sqa_acc,
        }

    print("=" * 104)
    print("  ASR = % FigStep judged UNSAFE by Llama Guard 3 (lower safer) | "
          "ORR = % benign over-refused (lower better) | SQA = ScienceQA acc (higher better)")
    print()

    out = os.path.join("results", "msr_corruptions_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  wrote %s\n" % out)


if __name__ == "__main__":
    main()
