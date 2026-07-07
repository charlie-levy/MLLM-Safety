#!/usr/bin/env python
"""
prep_sqa_test.py — download the FULL ScienceQA *test* split (paper-faithful) and
materialize it as offline files the GPU eval can read (compute nodes have no internet).

Run on the Newton LOGIN node (internet), REU env:
  conda activate REU
  export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python3 prep_sqa_test.py

Writes (default under /home/ch169788/experiments/part7/data):
    scienceqa_test.json          list of records:
        {idx, question, choices:[...], answer_idx:int, gold_letter:"A/B/...",
         hint, subject, grade, image_file (relative, or null if text-only)}
    sqa_test_images/<idx>.png    saved images (only for items that HAVE an image)

Source: derek-thomas/ScienceQA (test = 4,241). answer is an integer index into choices;
gold_letter = ['A','B','C','D','E'][answer]. Some test items have image=None (text-only) —
those are kept and evaluated WITHOUT an image (the paper evaluates the full test set).
"""
import os
import json
import argparse

OPTIONS = ["A", "B", "C", "D", "E"]


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

    img_dir = os.path.join(args.out_dir, "sqa_test_images")
    os.makedirs(img_dir, exist_ok=True)

    print("[prep_sqa] loading derek-thomas/ScienceQA split=test ...", flush=True)
    ds = load_dataset("derek-thomas/ScienceQA", split="test")
    print("[prep_sqa] %d test examples" % len(ds), flush=True)

    recs = []
    n_img = n_txt = 0
    for idx, ex in enumerate(ds):
        ans = ex["answer"]                      # int index into choices
        choices = list(ex["choices"])
        img = ex["image"]                       # PIL.Image or None
        image_file = None
        if img is not None:
            image_file = os.path.join("sqa_test_images", "%05d.png" % idx)
            img.convert("RGB").save(os.path.join(args.out_dir, image_file))
            n_img += 1
        else:
            n_txt += 1
        recs.append({
            "idx": idx,
            "question": ex["question"],
            "choices": choices,
            "answer_idx": int(ans),
            "gold_letter": OPTIONS[int(ans)],
            "hint": ex.get("hint", "") or "",
            "subject": ex.get("subject", ""),
            "grade": ex.get("grade", ""),
            "image_file": image_file,           # None for text-only items
        })
        if (idx + 1) % 500 == 0:
            print("  %d/%d  (img=%d txt=%d)" % (idx + 1, len(ds), n_img, n_txt), flush=True)

    out_json = os.path.join(args.out_dir, "scienceqa_test.json")
    json.dump(recs, open(out_json, "w"), ensure_ascii=False)

    print("\n=== DONE ===")
    print("  test records: %d  (with image: %d | text-only: %d)" % (len(recs), n_img, n_txt))
    print("  wrote: %s" % out_json)
    print("  images: %s" % img_dir)
    # sanity: gold letters distributed, all indices valid
    bad = [r["idx"] for r in recs if not (0 <= r["answer_idx"] < len(r["choices"]))]
    print("  answer_idx out of range: %d  (must be 0)" % len(bad))


if __name__ == "__main__":
    main()
