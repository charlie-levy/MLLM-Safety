#!/usr/bin/env python
"""
report_grid_guards.py — Plain ASR table for the model_responses grid.

Per cell (model x dataset x condition): how many responses each guard flagged
harmful, out of total -> ASR. Nothing else.

  python code/report_grid_guards.py results/grid_guard_eval
"""
import os
import sys
import json
import glob


def load_summaries(jdir):
    out = {}
    for p in glob.glob(os.path.join(jdir, "*.summary.json")):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        out[s["stem"]] = s
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/grid_guard_eval"
    lg = load_summaries(os.path.join(out_dir, "llamaguard"))
    wg = load_summaries(os.path.join(out_dir, "wildguard"))
    stems = sorted(set(lg) | set(wg))
    if not stems:
        sys.exit("no *.summary.json found under %s/{llamaguard,wildguard}" % out_dir)

    cells = []
    for stem in stems:
        ref = lg.get(stem) or wg.get(stem)
        cells.append({
            "model": ref.get("model", ""),
            "dataset": ref.get("dataset", ""),
            "condition": ref.get("condition", ""),
            "n_total": ref.get("n_total", 0),
            "n_harmful_llamaguard": lg.get(stem, {}).get("n_flag"),
            "asr_llamaguard": lg.get(stem, {}).get("asr_pct"),
            "n_harmful_wildguard": wg.get(stem, {}).get("n_flag"),
            "asr_wildguard": wg.get(stem, {}).get("asr_pct"),
        })

    def cell(asr, n):
        if asr is None:
            return "      -     "
        return "%6.2f%% (%d)" % (asr, n)

    print("\n  ASR = harmful responses / total\n")
    hdr = "  %-11s %-12s %-7s %5s |  %-13s %-13s" % (
        "model", "dataset", "cond", "N", "LLaMA-Guard", "WildGuard")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(cells, key=lambda x: (x["model"], x["dataset"], x["condition"])):
        print("  %-11s %-12s %-7s %5s |  %-13s %-13s" % (
            r["model"], r["dataset"], r["condition"], r["n_total"],
            cell(r["asr_llamaguard"], r["n_harmful_llamaguard"]),
            cell(r["asr_wildguard"], r["n_harmful_wildguard"])))

    sp = os.path.join(out_dir, "asr_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"cells": cells}, f, indent=2, ensure_ascii=False)
    print("\n  wrote %s\n" % sp)


if __name__ == "__main__":
    main()