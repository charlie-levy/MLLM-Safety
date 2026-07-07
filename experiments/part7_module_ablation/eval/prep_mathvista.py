#!/usr/bin/env python
"""
prep_mathvista.py — download MathVista *testmini* (1000) and materialize offline files
for the GPU eval (compute nodes have no internet).

Run on the Newton LOGIN node (internet), REU env:
  conda activate REU
  export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python3 prep_mathvista.py

Writes (default under /home/ch169788/experiments/part7/data):
    mathvista_testmini.json      list of records, one per problem, carrying EVERY field the
        official scorer needs: {pid, query, question, choices, unit, precision, answer,
        question_type, answer_type, metadata, image_file}
    mathvista_images/<pid>.png   the decoded image for each problem

Uses the dataset's built-in `query` field verbatim = the paper's exact prompt. `answer`,
`question_type`, `answer_type`, `precision`, `choices` are passed through unchanged so
extract_answer + calculate_score (official code) run exactly as in the paper.
"""
import os
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/ch169788/experiments/part7/data")
    args = ap.parse_args()

    # Force single-threaded arrow so parquet->arrow generation doesn't spawn a thread pool
    # (the Newton login node caps processes; compute nodes are fine). Harmless elsewhere.
    try:
        import pyarrow
        pyarrow.set_cpu_count(1)
        pyarrow.set_io_thread_count(1)
    except Exception:
        pass

    from datasets import load_dataset

    img_dir = os.path.join(args.out_dir, "mathvista_images")
    os.makedirs(img_dir, exist_ok=True)

    print("[prep_mv] loading AI4Math/MathVista split=testmini ...", flush=True)
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    print("[prep_mv] %d testmini examples" % len(ds), flush=True)

    recs = []
    for i, ex in enumerate(ds):
        pid = str(ex["pid"])
        img = ex["decoded_image"]                       # PIL.Image
        image_file = os.path.join("mathvista_images", "%s.png" % pid)
        img.convert("RGB").save(os.path.join(args.out_dir, image_file))
        recs.append({
            "pid": pid,
            "query": ex["query"],                        # built-in paper prompt (verbatim)
            "question": ex["question"],
            "choices": ex.get("choices"),                # list or None (free_form)
            "unit": ex.get("unit"),
            "precision": ex.get("precision"),
            "answer": ex["answer"],
            "question_type": ex["question_type"],        # multi_choice | free_form
            "answer_type": ex["answer_type"],            # integer | float | text | list
            "metadata": ex.get("metadata", {}),
            "image_file": image_file,
        })
        if (i + 1) % 200 == 0:
            print("  %d/%d" % (i + 1, len(ds)), flush=True)

    out_json = os.path.join(args.out_dir, "mathvista_testmini.json")
    json.dump(recs, open(out_json, "w"), ensure_ascii=False)

    qtypes = {}
    for r in recs:
        qtypes[r["question_type"]] = qtypes.get(r["question_type"], 0) + 1
    print("\n=== DONE ===")
    print("  testmini records: %d   question_types=%s" % (len(recs), qtypes))
    print("  wrote:  %s" % out_json)
    print("  images: %s" % img_dir)
    missing = [r["pid"] for r in recs if not os.path.isfile(os.path.join(args.out_dir, r["image_file"]))]
    print("  images missing on disk: %d  (must be 0)" % len(missing))


if __name__ == "__main__":
    main()
