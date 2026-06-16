#!/usr/bin/env python
"""
format_orr_results.py — Merge ORR judge verdicts into the keyed-by-idx house format.

judge_safety_hf.py --mode orr scores responses_orr.csv and writes
judged_llama_orr.csv (per-sample verdicts) + judged_llama_orr.json (aggregate).
This step joins those verdicts back onto the per-sample RESPONSE records (which
carry image_path + label) and writes, per dataset, a keyed-by-idx JSON matching
the FigStep/ASR output and the advisor's house format:

  <cond_dir>/xstest_results.json   { "<idx>": { idx, dataset, prompt, image_path,
       label, full_response, judge_input, judge_raw_output, verdict, flag } }
  <cond_dir>/mmsa_results.json     same schema

Pure post-processing (no model) — fast, runs right after the ORR judge in the
same job.

Usage:
  python code/format_orr_results.py results/msr_guard_eval/clean
"""
import os
import sys
import csv
import json
import argparse

csv.field_size_limit(10_000_000)   # judge_input/full_response can be long

DATASETS = {"xstest": "XSTest", "mmsa": "MMSA"}


def load_keyed(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_judged_rows(csv_path):
    """Return {dataset_upper: {idx_str: judged_row}} from judged_llama_orr.csv."""
    by_ds = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = str(row.get("dataset", "")).upper()
            by_ds.setdefault(ds, {})[str(row.get("idx", ""))] = row
    return by_ds


def main():
    ap = argparse.ArgumentParser(description="Format ORR judge output to keyed-by-idx JSON")
    ap.add_argument("cond_dir", help="e.g. results/msr_guard_eval/clean")
    args = ap.parse_args()

    judged_csv = os.path.join(args.cond_dir, "judged_llama_orr.csv")
    if not os.path.exists(judged_csv):
        raise FileNotFoundError("missing %s — run judge_safety_hf.py --mode orr first" % judged_csv)
    judged = load_judged_rows(judged_csv)

    for short, ds_name in DATASETS.items():
        resp_path = os.path.join(args.cond_dir, "responses_%s.json" % short)
        if not os.path.exists(resp_path):
            raise FileNotFoundError("missing %s" % resp_path)
        responses = load_keyed(resp_path)
        verdicts = judged.get(ds_name.upper(), {})

        out = {}
        missing_verdict = []
        for key, rec in responses.items():
            v = verdicts.get(str(key))
            if v is None:
                missing_verdict.append(key)
                continue
            out[key] = {
                "idx":            rec.get("idx", key),
                "dataset":        ds_name,
                "prompt":         rec.get("prompt", ""),
                "image_path":     rec.get("image_path", ""),
                "label":          rec.get("label", ""),
                "full_response":  rec.get("full_response", ""),
                "judge_input":    v.get("judge_input", ""),     # EXACT text shown to judge
                "judge_raw_output": v.get("judge_raw", ""),
                "verdict":        v.get("verdict", ""),
                "flag":           int(v.get("flag", 0)),        # 1 = over-refusal
            }

        if missing_verdict:
            # Loud failure: every response must have a verdict (no silent drops).
            raise RuntimeError(
                "%s: %d response rows had no matching judged verdict (idx %s ...)"
                % (ds_name, len(missing_verdict), missing_verdict[:5]))

        out_path = os.path.join(args.cond_dir, "%s_results.json" % short)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        n_orr = sum(r["flag"] for r in out.values())
        print("  %-6s -> %s  (%d samples, %d over-refusals)"
              % (ds_name, out_path, len(out), n_orr), flush=True)


if __name__ == "__main__":
    main()
