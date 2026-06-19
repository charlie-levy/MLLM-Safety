#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combine_unsafe.py — merge the judged conditions into ONE neat results file.

Reads the per-condition judged_<tag>.json + asr_<tag>.json and writes a single
unsafe_6_19/RESULTS_unsafe.json where each image shows the FULL response + judge
verdict for every condition side by side, plus a summary ASR table. Also prints
the table. CPU-only (no GPU).

  python unsafe_6_19/combine_unsafe.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
DIR = "unsafe_6_19"

# (column label, response/judged tag).  base only clean; TIS clean/20/40.
CONDITIONS = [
    ("base_clean",  "base_clean"),
    ("tis_clean",   "tis_clean"),
    ("tis_blur20",  "tis_blur20"),
    ("tis_blur40",  "tis_blur40"),
]


def _load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def main():
    summary, judged = {}, {}
    for col, tag in CONDITIONS:
        judged[col] = _load(os.path.join(DIR, "judged_%s.json" % tag))
        agg = _load(os.path.join(DIR, "asr_%s.json" % tag))
        summary[col] = ({"asr_pct": agg["asr_pct"], "n_unsafe": agg["n_unsafe"],
                         "n_total": agg["n_total"]} if agg else None)

    # union of all idx seen across conditions, sorted numerically
    idxs = set()
    for col, _ in CONDITIONS:
        if judged[col]:
            idxs.update(judged[col].keys())
    idxs = sorted(idxs, key=lambda x: int(x))

    samples = {}
    for k in idxs:
        ref = next((judged[c][k] for c, _ in CONDITIONS if judged[c] and k in judged[c]), {})
        entry = {"idx": ref.get("idx", k), "image_path": ref.get("image_path", ""),
                 "prompt": ref.get("prompt", "")}
        for col, _ in CONDITIONS:
            rec = judged[col].get(k) if judged[col] else None
            entry[col] = ({"verdict": rec["verdict"], "attack_success": rec["flag"],
                           "full_response": rec["full_response"],
                           "judge_raw_output": rec["judge_raw_output"]} if rec else None)
        samples[k] = entry

    out = {"summary": summary, "samples": samples}
    outp = os.path.join(DIR, "RESULTS_unsafe.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print("  Unsafe-images ASR (Llama Guard 3 Vision; unsafe response = success)")
    print("=" * 64)
    print("  %-12s  %8s  %s" % ("Condition", "ASR", "unsafe / total"))
    print("  " + "-" * 60)
    for col, _ in CONDITIONS:
        s = summary[col]
        if s:
            print("  %-12s  %7.2f%%  %d / %d" % (col, s["asr_pct"], s["n_unsafe"], s["n_total"]))
        else:
            print("  %-12s  %8s  %s" % (col, "n/a", "(not run yet)"))
    print("=" * 64)
    print("  wrote %s  (%d images)\n" % (outp, len(samples)))


if __name__ == "__main__":
    main()
