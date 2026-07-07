#!/usr/bin/env python
"""
prep_mathvista.py — materialize MathVista *testmini* (1000) as offline files for the GPU eval.

Reads the testmini parquet DIRECTLY with single-threaded pyarrow (use_threads=False) — no
`datasets` ThreadPoolExecutor (which hits the Newton login-node process cap), no OOM. Same data
(AI4Math/MathVista testmini), just slower.

  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python3 prep_mathvista.py

Writes (default under /home/ch169788/experiments/part7/data):
    mathvista_testmini.json      {pid, query, question, choices, unit, precision, answer,
                                  question_type, answer_type, metadata, image_file}
    mathvista_images/<pid>.png   the decoded image for each problem

Uses the built-in `query` field verbatim (the paper's exact prompt); the rest pass through
unchanged for the official scorer.
"""
import os
import io
import glob
import json
import argparse

MV_TESTMINI_GLOB = ("~/.cache/huggingface/hub/datasets--AI4Math--MathVista/"
                    "snapshots/*/data/testmini-*.parquet")
NEEDED = ["pid", "question", "decoded_image", "choices", "unit", "precision",
          "answer", "question_type", "answer_type", "metadata", "query"]


def to_pil(val):
    """MathVista 'decoded_image' comes from parquet as a {'bytes','path'} struct."""
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
    ap.add_argument("--parquet_glob", default=MV_TESTMINI_GLOB)
    args = ap.parse_args()

    import pyarrow
    import pyarrow.parquet as pq
    pyarrow.set_cpu_count(1)

    files = sorted(glob.glob(os.path.expanduser(args.parquet_glob)))
    if not files:
        raise SystemExit("MathVista testmini parquet not found in HF cache:\n  %s"
                         % os.path.expanduser(args.parquet_glob))
    print("[prep_mv] reading testmini parquet(s) single-threaded:\n  %s" % "\n  ".join(files), flush=True)
    tabs = [pq.read_table(f, use_threads=False) for f in files]
    table = pyarrow.concat_tables(tabs) if len(tabs) > 1 else tabs[0]
    n = table.num_rows
    cols = {c: table.column(c) for c in table.column_names}
    print("[prep_mv] %d testmini examples" % n, flush=True)

    def get(c, i):
        return cols[c][i].as_py() if c in cols else None

    img_dir = os.path.join(args.out_dir, "mathvista_images")
    os.makedirs(img_dir, exist_ok=True)

    recs = []
    for i in range(n):
        pid = str(get("pid", i))
        img = to_pil(get("decoded_image", i))
        image_file = os.path.join("mathvista_images", "%s.png" % pid)
        img.convert("RGB").save(os.path.join(args.out_dir, image_file))
        ch = get("choices", i)
        recs.append({
            "pid": pid,
            "query": get("query", i),
            "question": get("question", i),
            "choices": list(ch) if ch is not None else None,
            "unit": get("unit", i),
            "precision": get("precision", i),
            "answer": get("answer", i),
            "question_type": get("question_type", i),
            "answer_type": get("answer_type", i),
            "metadata": get("metadata", i) or {},
            "image_file": image_file,
        })
        if (i + 1) % 200 == 0:
            print("  %d/%d" % (i + 1, n), flush=True)

    out_json = os.path.join(args.out_dir, "mathvista_testmini.json")
    json.dump(recs, open(out_json, "w"), ensure_ascii=False)

    qtypes = {}
    for r in recs:
        qtypes[r["question_type"]] = qtypes.get(r["question_type"], 0) + 1
    missing = [r["pid"] for r in recs if not os.path.isfile(os.path.join(args.out_dir, r["image_file"]))]
    print("\n=== DONE ===")
    print("  testmini records: %d   question_types=%s" % (len(recs), qtypes))
    print("  wrote:  %s" % out_json)
    print("  images: %s" % img_dir)
    print("  images missing on disk: %d  (must be 0)" % len(missing))


if __name__ == "__main__":
    main()
