#!/usr/bin/env python
"""
jsonl_to_official.py — convert our responses_<model>.jsonl into the OFFICIAL MathVista
results format so the files load straight into lupantech/MathVista's
evaluation/extract_answer.py + calculate_score.py with zero edits.

Official format = a single JSON object keyed by string pid, each value being the problem
dict + `query` + `response`:
    { "1": {"pid":"1","question":...,"choices":...,"unit":...,"precision":...,"answer":...,
            "question_type":...,"answer_type":...,"metadata":{...},"query":...,"response":...},
      "2": {...}, ... }
extract_answer.py adds "extraction"; calculate_score.py adds "true_false".

  python jsonl_to_official.py --dir <dir with responses_*.jsonl>
-> writes results_<model>.json next to each responses_<model>.jsonl
"""
import os
import json
import glob
import argparse

# every field the official problem dict carries; extract_answer.py/calculate_score.py read a
# subset (query, response, choices, precision, answer, question_type, answer_type, metadata).
OFFICIAL_KEYS = ["pid", "question", "choices", "unit", "precision", "answer",
                 "question_type", "answer_type", "metadata", "query", "response"]


def convert(path):
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        pid = str(r["pid"])
        out[pid] = {k: r[k] for k in OFFICIAL_KEYS if k in r}
        out[pid]["pid"] = pid
    outp = os.path.join(os.path.dirname(path),
                        "results_" + os.path.basename(path).replace("responses_", "").replace(".jsonl", ".json"))
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=2)
    have = sorted(set().union(*[set(v) for v in out.values()])) if out else []
    print("  %s -> %s  (%d problems; fields=%s)"
          % (os.path.basename(path), os.path.basename(outp), len(out), have))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    paths = list(args.files) + sorted(glob.glob(os.path.join(args.dir, "responses_*.jsonl")))
    if not paths:
        raise SystemExit("no responses_*.jsonl in %s" % args.dir)
    print("Converting to official MathVista results format:")
    for p in paths:
        convert(p)


if __name__ == "__main__":
    main()
