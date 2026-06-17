#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_msr_multirun.py — Average MSR-Align results across N independent eval runs.

Reads results/msr_guard_eval_run{1..N}/{clean,blur20}/
  asr_guard.json        (ASR from Llama Guard 3 Vision)
  judged_llama_orr.json (ORR from LLaMA-3-8B)

Computes mean +/- std across runs, prints a table, writes:
  results/msr_guard_multirun_summary.json

SQA (LLaMA-judged) is pulled from the existing single-run dirs — not replicated
per run since it is deterministic and already confirmed.

CPU-only — safe on the login node.

Usage:
  python code/report_msr_multirun.py [--n_runs 3]
"""
import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONDITIONS = ["clean", "blur20", "blur40"]

SQA_FILES = {
    "clean":  os.path.join("results", "sqa_noise_sweep",
                           "judged_base_msr_clean.json"),
    "blur20": os.path.join("results", "sqa_blur_pct",
                           "judged_base_msr_gaussian_blur_pct_p20.json"),
    "blur40": os.path.join("results", "sqa_blur_pct",
                           "judged_base_msr_gaussian_blur_pct_p40.json"),
}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def _std(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def collect(n_runs):
    """Return {cond: {metric: [val_run1, val_run2, ...]}} and found-count per cond."""
    raw = {c: {"asr": [], "xs_orr": [], "mm_orr": [], "avg_orr": []}
           for c in CONDITIONS}
    found = {c: 0 for c in CONDITIONS}

    for r in range(1, n_runs + 1):
        for cond in CONDITIONS:
            d = os.path.join("results", "msr_guard_eval_run%d" % r, cond)
            asr = _load(os.path.join(d, "asr_guard.json"))
            orr = _load(os.path.join(d, "judged_llama_orr.json"))
            if asr is None or orr is None:
                missing = []
                if asr is None: missing.append("asr_guard.json")
                if orr is None: missing.append("judged_llama_orr.json")
                print("  [!] run %d %s: missing %s" % (r, cond, ", ".join(missing)))
                continue
            found[cond] += 1
            raw[cond]["asr"].append(asr.get("asr_pct"))
            raw[cond]["xs_orr"].append(
                orr.get("xstest", {}).get("orr_pct"))
            raw[cond]["mm_orr"].append(
                orr.get("mmsa", {}).get("orr_pct"))
            raw[cond]["avg_orr"].append(orr.get("avg_orr_pct"))

    return raw, found


def fmt_ms(m, s):
    """Format mean +/- std; n/a if missing."""
    if m is None:
        return "      n/a     "
    return "%6.2f%% +/-%5.2f" % (m, s)


def fmt_v(v):
    return ("%6.2f%%" % v) if v is not None else "  n/a  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=3,
                    help="Number of independent runs to average (default 3)")
    args = ap.parse_args()

    raw, found = collect(args.n_runs)

    sqa = {}
    for c in CONDITIONS:
        d = _load(SQA_FILES.get(c, ""))
        sqa[c] = d.get("accuracy") if d else None

    col = 17
    print()
    print("=" * 100)
    print("  MSR-Align Guard Eval  |  %d-run mean +/- std  "
          "|  Llama Guard 3 ASR  +  LLaMA-3 ORR  +  LLaMA SQA" % args.n_runs)
    print("=" * 100)
    print("  %-8s  %-18s  %-18s  %-18s  %-18s  %8s  %s" % (
        "Cond", "ASR (Guard) [sfer<]", "ORR XSTest [<btr]",
        "ORR MMSA [<btr]", "ORR Avg [<btr]", "SQA [>]", "n_runs"))
    print("  " + "-" * 96)

    summary = {}
    for cond in CONDITIONS:
        d = raw[cond]
        asr_m = _mean(d["asr"]);   asr_s = _std(d["asr"])
        xs_m  = _mean(d["xs_orr"]); xs_s = _std(d["xs_orr"])
        mm_m  = _mean(d["mm_orr"]); mm_s = _std(d["mm_orr"])
        av_m  = _mean(d["avg_orr"]); av_s = _std(d["avg_orr"])

        print("  %-8s  %-18s  %-18s  %-18s  %-18s  %8s  %d/%d" % (
            cond,
            fmt_ms(asr_m, asr_s),
            fmt_ms(xs_m,  xs_s),
            fmt_ms(mm_m,  mm_s),
            fmt_ms(av_m,  av_s),
            fmt_v(sqa[cond]),
            found[cond], args.n_runs))

        summary[cond] = {
            "n_runs_completed": found[cond],
            "asr_pct":     {"mean": asr_m, "std": asr_s, "raw": d["asr"]},
            "xstest_orr":  {"mean": xs_m,  "std": xs_s,  "raw": d["xs_orr"]},
            "mmsa_orr":    {"mean": mm_m,  "std": mm_s,  "raw": d["mm_orr"]},
            "avg_orr":     {"mean": av_m,  "std": av_s,  "raw": d["avg_orr"]},
            "sqa_acc":     sqa[cond],
        }

    print("=" * 100)
    print("  ASR  (Guard) = %% FigStep labeled UNSAFE by Llama Guard 3-11B-Vision  (lower = safer)")
    print("  ORR  (LLaMA) = %% benign XSTest/MMSA prompts over-refused              (lower = better)")
    print("  SQA  (LLaMA) = %% ScienceQA accuracy, single-run, LLaMA-3-8B judge    (higher = better)")
    print()

    incomplete = [c for c in CONDITIONS if found[c] < args.n_runs]
    if incomplete:
        print("  [!] incomplete runs for: %s — re-check squeue or logs\n" %
              ", ".join(incomplete))

    out = os.path.join("results", "msr_guard_multirun_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  wrote %s\n" % out)


if __name__ == "__main__":
    main()
