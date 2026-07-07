#!/usr/bin/env python
"""
prep_sqa_test.py — materialize the FULL ScienceQA *test* split (paper-faithful) as offline
files the GPU eval can read.

Reads the test parquet DIRECTLY with single-threaded pyarrow (use_threads=False). This avoids
(a) load_dataset regenerating the huge train/val splits (OOM) and (b) the `datasets` ArrowReader
ThreadPoolExecutor (which hits the Newton login-node process cap: "can't start new thread").
Pure pyarrow spawns no threads — same data (derek-thomas/ScienceQA test, 4,241), just slower.

  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python3 prep_sqa_test.py

Writes (default under /home/ch169788/experiments/part7/data):
    scienceqa_test.json          {idx, question, choices, answer_idx, gold_letter, hint,
                                  subject, grade, image_file (or null if text-only)}
    sqa_test_images/<idx>.png    images (only for items that have one)

answer = int index into choices; gold_letter = ['A'..'E'][answer]. Text-only items (image=None)
are kept (the paper evals the full test set).
"""
import os
import io
import glob
import json
import argparse

OPTIONS = ["A", "B", "C", "D", "E"]
SQA_TEST_GLOB = ("~/.cache/huggingface/hub/datasets--derek-thomas--ScienceQA/"
                 "snapshots/*/data/test-*.parquet")
NEEDED = ["image", "question", "choices", "answer", "hint", "subject", "grade"]


def to_pil(val):
    """ScienceQA 'image' comes from parquet as a {'bytes','path'} struct (or None)."""
    from PIL import Image
    if val is None:
        return None
    if hasattr(val, "convert"):
        return val
    if isinstance(val, dict) and val.get("bytes"):
        return Image.open(io.BytesIO(val["bytes"]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/ch169788/experiments/part7/data")
    ap.add_argument("--parquet_glob", default=SQA_TEST_GLOB)
    args = ap.parse_args()

    import pyarrow
    import pyarrow.parquet as pq
    pyarrow.set_cpu_count(1)

    files = sorted(glob.glob(os.path.expanduser(args.parquet_glob)))
    if not files:
        raise SystemExit("ScienceQA test parquet not found in HF cache:\n  %s"
                         % os.path.expanduser(args.parquet_glob))
    print("[prep_sqa] reading test parquet(s) single-threaded:\n  %s" % "\n  ".join(files), flush=True)
    tabs = [pq.read_table(f, use_threads=False) for f in files]
    table = pyarrow.concat_tables(tabs) if len(tabs) > 1 else tabs[0]
    n = table.num_rows
    cols = {c: table.column(c) for c in table.column_names}
    print("[prep_sqa] %d test examples" % n, flush=True)

    def get(c, i):
        return cols[c][i].as_py() if c in cols else None

    img_dir = os.path.join(args.out_dir, "sqa_test_images")
    os.makedirs(img_dir, exist_ok=True)

    recs = []
    n_img = n_txt = 0
    for idx in range(n):
        img = to_pil(get("image", idx))
        image_file = None
        if img is not None:
            image_file = os.path.join("sqa_test_images", "%05d.png" % idx)
            img.convert("RGB").save(os.path.join(args.out_dir, image_file))
            n_img += 1
        else:
            n_txt += 1
        ans = int(get("answer", idx))
        recs.append({
            "idx": idx,
            "question": get("question", idx),
            "choices": list(get("choices", idx)),
            "answer_idx": ans,
            "gold_letter": OPTIONS[ans],
            "hint": get("hint", idx) or "",
            "subject": get("subject", idx) or "",
            "grade": get("grade", idx) or "",
            "image_file": image_file,
        })
        if (idx + 1) % 500 == 0:
            print("  %d/%d  (img=%d txt=%d)" % (idx + 1, n, n_img, n_txt), flush=True)

    out_json = os.path.join(args.out_dir, "scienceqa_test.json")
    json.dump(recs, open(out_json, "w"), ensure_ascii=False)

    bad = [r["idx"] for r in recs if not (0 <= r["answer_idx"] < len(r["choices"]))]
    print("\n=== DONE ===")
    print("  test records: %d  (with image: %d | text-only: %d)" % (len(recs), n_img, n_txt))
    print("  wrote: %s" % out_json)
    print("  images: %s" % img_dir)
    print("  answer_idx out of range: %d  (must be 0)" % len(bad))


if __name__ == "__main__":
    main()