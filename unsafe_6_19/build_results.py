#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_results.py — merge the 4 judged conditions into ONE flat results file.

Reads judged_<condition>[_pilot].json for the 4 conditions and writes a flat list
of entries (one per image x condition) in the exact checklist structure. CPU-only.

  python unsafe_6_19/build_results.py            # -> unsafe100_asr_results.json
  python unsafe_6_19/build_results.py --pilot    # -> pilot_output.json (+ prints each)

Each entry:
  {image_index, image_path, condition, prompt, model_response,
   judge_prompt_sent, judge_response_full, judge_verdict, is_attack_success}
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
DIR = "unsafe_6_19"

CONDITIONS = ["base_clean", "tis_clean", "tis_blur_20", "tis_blur_40"]
FIELDS = ["image_index", "image_path", "condition", "prompt", "model_response",
          "judge_prompt_sent", "judge_response_full", "judge_verdict", "is_attack_success"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    suffix = "_pilot" if args.pilot else ""
    out_name = "pilot_output.json" if args.pilot else "unsafe100_asr_results.json"

    entries = []
    for cond in CONDITIONS:
        p = os.path.join(DIR, "judged_%s%s.json" % (cond, suffix))
        if not os.path.exists(p):
            print("  [skip] %s not found (condition not judged yet)" % p)
            continue
        data = json.load(open(p, encoding="utf-8"))
        for k in sorted(data, key=lambda x: int(x)):
            r = data[k]
            entries.append({f: r.get(f) for f in FIELDS})

    out_path = os.path.join(DIR, out_name)
    json.dump(entries, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote %s  (%d entries)" % (out_path, len(entries)))

    if args.pilot:
        print("\n" + "=" * 70)
        print("  PILOT — read these 4 and confirm the judge is working:")
        print("=" * 70)
        for e in entries:
            resp = (e["model_response"] or "").replace("\n", " ")
            print("\n  condition : %s" % e["condition"])
            print("  verdict   : %s   (attack_success=%s)" % (e["judge_verdict"], e["is_attack_success"]))
            print("  response  : %s%s" % (resp[:200], "..." if len(resp) > 200 else ""))
            print("  judge_raw : %r" % e["judge_response_full"])


if __name__ == "__main__":
    main()
