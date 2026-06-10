#!/usr/bin/env python
"""
jsonl_to_csv.py — Convert raw_*.jsonl SQA response files to full-response CSV.

Reads every raw_*.jsonl under a directory (produced by eval_sqa_noise_sweep.py)
and writes a matching responses_*.csv with the complete model output.

Usage:
  python code/jsonl_to_csv.py --dir results_newton/sqa_noise_sweep
  python code/jsonl_to_csv.py --dir results_newton/sqa_blur_sweep
  python code/jsonl_to_csv.py results_newton/sqa_noise_sweep/raw_base_clean.jsonl
"""
import csv, glob, json, os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

from metrics import extract_answer_letter, is_refusal, _extract_conclusion

FIELDS = [
    "idx", "prompt", "label",
    "conclusion", "full_response",
    "predicted_letter", "correct",
    "is_refusal",
]


def convert(path):
    with open(path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    out = os.path.join(os.path.dirname(path),
                       os.path.basename(path).replace("raw_", "responses_").replace(".jsonl", ".csv"))

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for it in items:
            resp   = it.get("response", "")
            conc   = (_extract_conclusion(resp) or "").strip()
            pred   = extract_answer_letter(resp)
            lbl    = it.get("label", "").strip().upper()
            correct = (pred == lbl) if pred and lbl else None
            writer.writerow({
                "idx":              it.get("idx", ""),
                "prompt":           it.get("prompt", ""),
                "label":            it.get("label", ""),
                "conclusion":       conc,
                "full_response":    resp,
                "predicted_letter": pred or "",
                "correct":          "" if correct is None else correct,
                "is_refusal":       is_refusal(resp),
            })

    n = len(items)
    print("  %-45s  %d rows  ->  %s" % (os.path.basename(path), n, os.path.basename(out)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="raw_*.jsonl files")
    ap.add_argument("--dir", help="convert every raw_*.jsonl in this directory")
    args = ap.parse_args()

    paths = list(args.files)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "raw_*.jsonl")))
    if not paths:
        sys.exit("No input. Pass raw_*.jsonl files or --dir <folder>.")

    print("Converting %d file(s)...\n" % len(paths))
    for p in paths:
        convert(p)
    print("\nDone.")


if __name__ == "__main__":
    main()
