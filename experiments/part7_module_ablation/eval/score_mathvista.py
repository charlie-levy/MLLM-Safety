#!/usr/bin/env python
"""
score_mathvista.py — MathVista testmini accuracy using the OFFICIAL scoring code.

Imports normalize_extracted_answer + safe_equal VERBATIM from the cloned MathVista repo
(nothing reimplemented), so scoring is exactly the paper's. Reports overall accuracy plus
a breakdown by question_type and answer_type.

Requires the clone + its scoring dep:
    git clone --depth 1 https://github.com/lupantech/MathVista ~/MathVista
    pip install python-Levenshtein          # used by the official multi_choice normalizer

  conda activate REU
  python score_mathvista.py --dir /home/ch169788/experiments/part7/mathvista
  python score_mathvista.py /home/ch169788/experiments/part7/mathvista/extracted_llm.jsonl
"""
import os
import sys
import json
import glob
import argparse


def load_official(mathvista_repo):
    repo = os.path.expanduser(mathvista_repo)
    evald = os.path.join(repo, "evaluation")
    if not os.path.isdir(evald):
        raise SystemExit("MathVista repo not found at %s (git clone --depth 1 "
                         "https://github.com/lupantech/MathVista ~/MathVista)" % mathvista_repo)
    # calculate_score.py does `from utilities import ...`; utilities.py lives at the repo root
    # (not in evaluation/), so put BOTH the repo root and evaluation/ on sys.path.
    sys.path.insert(0, repo)
    sys.path.insert(0, evald)
    from calculate_score import normalize_extracted_answer, safe_equal   # noqa
    return normalize_extracted_answer, safe_equal


def score_file(path, normalize, safe_equal):
    items = [json.loads(l) for l in open(path) if l.strip()]
    correct = 0
    by_qt = {}      # question_type -> [correct, total]
    by_at = {}      # answer_type   -> [correct, total]
    graded = []
    for it in items:
        norm = normalize(
            it.get("extraction", ""), it.get("choices"),
            it["question_type"], it["answer_type"], it.get("precision"))
        ok = bool(safe_equal(norm, it["answer"]))
        correct += int(ok)
        for d, k in ((by_qt, it["question_type"]), (by_at, it["answer_type"])):
            d.setdefault(k, [0, 0])
            d[k][1] += 1
            d[k][0] += int(ok)
        graded.append({"pid": it["pid"], "normalized": norm, "answer": it["answer"], "true_false": ok})
    total = len(items)
    acc = 100.0 * correct / total if total else 0.0
    tag = os.path.basename(path).replace("extracted_", "").replace(".jsonl", "")
    out = {
        "tag": tag,
        "metric": "MathVista testmini accuracy (official scoring; Llama-3-8B extractor)",
        "accuracy": round(acc, 2),
        "correct": correct,
        "total": total,
        "by_question_type": {k: round(100.0 * v[0] / v[1], 2) for k, v in sorted(by_qt.items())},
        "by_answer_type": {k: round(100.0 * v[0] / v[1], 2) for k, v in sorted(by_at.items())},
        "graded": graded,
    }
    outp = os.path.join(os.path.dirname(path), "scores_%s.json" % tag)
    json.dump(out, open(outp, "w"), indent=2)
    print("  %-8s accuracy=%5.2f%%  (%d/%d)  by_qtype=%s  -> %s"
          % (tag, acc, correct, total, out["by_question_type"], os.path.basename(outp)))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir")
    ap.add_argument("--mathvista_repo", default="~/MathVista")
    args = ap.parse_args()
    paths = list(args.files)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "extracted_*.jsonl")))
    if not paths:
        sys.exit("No input. Pass extracted_*.jsonl or --dir.")
    normalize, safe_equal = load_official(args.mathvista_repo)
    print("=== MathVista testmini accuracy (official scoring) ===")
    for p in paths:
        score_file(p, normalize, safe_equal)


if __name__ == "__main__":
    main()
