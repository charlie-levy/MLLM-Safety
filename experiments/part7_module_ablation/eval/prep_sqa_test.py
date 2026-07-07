#!/usr/bin/env python
"""
prep_sqa_test.py — materialize the FULL ScienceQA *test* split (paper-faithful) as offline
files the GPU eval can read (compute nodes have no internet).

Loads ONLY the test parquet directly from the HF hub cache (populated by the earlier
`hf`/load_dataset download). This deliberately avoids load_dataset("...ScienceQA", split="test"),
which regenerates the huge train+val splits too and OOMs the login node. Reading one parquet
also needs no hub/offline/cache-builder logic, so it runs anywhere.

Run in the REU env (login or compute node):
  export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python3 prep_sqa_test.py

Writes (default under /home/ch169788/experiments/part7/data):
    scienceqa_test.json          {idx, question, choices, answer_idx, gold_letter, hint,
                                  subject, grade, image_file (or null if text-only)}
    sqa_test_images/<idx>.png    images (only for items that have one)

Source parquet: derek-thomas/ScienceQA test (4,241). answer = int index into choices;
gold_letter = ['A'..'E'][answer]. Text-only items (image=None) are kept (paper evals the full test).
"""
import os
import io
import glob
import json
import argparse

OPTIONS = ["A", "B", "C", "D", "E"]
SQA_TEST_GLOB = ("~/.cache/huggingface/hub/datasets--derek-thomas--ScienceQA/"
                 "snapshots/*/data/test-*.parquet")


def to_pil(val):
    """ScienceQA 'image' is a HF Image; from a raw parquet it's a {'bytes','path'} dict."""
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

    try:
        import pyarrow
        pyarrow.set_cpu_count(1)
        pyarrow.set_io_thread_count(1)
    except Exception:
        pass
    from datasets import load_dataset

    files = sorted(glob.glob(os.path.expanduser(args.parquet_glob)))
    if not files:
        raise SystemExit("ScienceQA test parquet not found in HF cache:\n  %s\n"
                         "(download it first with: python -c \"from datasets import load_dataset; "
                         "load_dataset('derek-thomas/ScienceQA', split='test')\" on the login node)"
                         % os.path.expanduser(args.parquet_glob))
    print("[prep_sqa] loading test parquet(s) directly:\n  %s" % "\n  ".join(files), flush=True)
    ds = load_dataset("parquet", data_files=files, split="train")   # single file -> 'train' handle
    print("[prep_sqa] %d test examples" % len(ds), flush=True)

    img_dir = os.path.join(args.out_dir, "sqa_test_images")
    os.makedirs(img_dir, exist_ok=True)

    recs = []
    n_img = n_txt = 0
    for idx, ex in enumerate(ds):
        img = to_pil(ex["image"])
        image_file = None
        if img is not None:
            image_file = os.path.join("sqa_test_images", "%05d.png" % idx)
            img.convert("RGB").save(os.path.join(args.out_dir, image_file))
            n_img += 1
        else:
            n_txt += 1
        ans = int(ex["answer"])
        recs.append({
            "idx": idx,
            "question": ex["question"],
            "choices": list(ex["choices"]),
            "answer_idx": ans,
            "gold_letter": OPTIONS[ans],
            "hint": ex.get("hint", "") or "",
            "subject": ex.get("subject", ""),
            "grade": ex.get("grade", ""),
            "image_file": image_file,
        })
        if (idx + 1) % 500 == 0:
            print("  %d/%d  (img=%d txt=%d)" % (idx + 1, len(ds), n_img, n_txt), flush=True)

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