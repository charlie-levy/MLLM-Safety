#!/usr/bin/env python
"""
build_judge_dataset.py — turn the HoliSafe-SSU manifest into the dataset JSON the
SIUO R/C Conclusion judge (judging/eval_hr_table1_siuo_jsonl_R_C.py) expects, so
that judge runs on SSU responses UNCHANGED.

The SIUO judge does:  lookup = {int(v["question_id"]): v for v in raw.values()}
and pulls `category` + `image_path` per item; `safety_warning`/`reference_answer`
are intentionally omitted so it uses its GENERIC safety fallback (the agreed
"generic SIUO rubric + category"). image_path is the CLEAN SSU image (correct:
the judge scores the response against the true scenario, as in Parts 4/5/8/10).

Run on Newton, then point the judge at --siuo-dataset <out> --images-dir "".
"""
import os
import json
import argparse

MANIFEST = "/home/ch169788/experiments/part2/data/holisafe_ssu/manifest.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default="/home/ch169788/judging/holisafe_ssu_judge_dataset.json")
    args = ap.parse_args()

    out = {}
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            qid = m["idx"]
            out[str(qid)] = {
                "question_id": qid,
                "category": m.get("category", ""),
                "image_path": m["image_path"],
            }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s with %d items" % (args.out, len(out)))


if __name__ == "__main__":
    main()
