#!/usr/bin/env python
"""
prep_mathvista.py — materialize MathVista *testmini* (1000) as offline files for the GPU eval.

Loads ONLY the testmini parquet directly from the HF hub cache (avoids the hub/offline/
cache-builder path and any train/test regeneration). Runs anywhere in the REU env.

  export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python3 prep_mathvista.py

Writes (default under /home/ch169788/experiments/part7/data):
    mathvista_testmini.json      {pid, query, question, choices, unit, precision, answer,
                                  question_type, answer_type, metadata, image_file}
    mathvista_images/<pid>.png   the decoded image for each problem

Uses the built-in `query` field verbatim (the paper's exact prompt); answer/question_type/
answer_type/precision/choices pass through unchanged for the official scorer.
"""
import os
import io
import glob
import json
import argparse

MV_TESTMINI_GLOB = ("~/.cache/huggingface/hub/datasets--AI4Math--MathVista/"
                    "snapshots/*/data/testmini-*.parquet")


def to_pil(val):
    """MathVista 'decoded_image' is a HF Image; from a raw parquet it's a {'bytes','path'} dict."""
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

    try:
        import pyarrow
        pyarrow.set_cpu_count(1)
        pyarrow.set_io_thread_count(1)
    except Exception:
        pass
    from datasets import load_dataset

    files = sorted(glob.glob(os.path.expanduser(args.parquet_glob)))
    if not files:
        raise SystemExit("MathVista testmini parquet not found in HF cache:\n  %s\n"
                         "(download it first with: python -c \"from datasets import load_dataset; "
                         "load_dataset('AI4Math/MathVista', split='testmini')\" on the login node)"
                         % os.path.expanduser(args.parquet_glob))
    print("[prep_mv] loading testmini parquet(s) directly:\n  %s" % "\n  ".join(files), flush=True)
    ds = load_dataset("parquet", data_files=files, split="train")
    print("[prep_mv] %d testmini examples" % len(ds), flush=True)

    img_dir = os.path.join(args.out_dir, "mathvista_images")
    os.makedirs(img_dir, exist_ok=True)

    recs = []
    for i, ex in enumerate(ds):
        pid = str(ex["pid"])
        img = to_pil(ex["decoded_image"])
        image_file = os.path.join("mathvista_images", "%s.png" % pid)
        img.convert("RGB").save(os.path.join(args.out_dir, image_file))
        recs.append({
            "pid": pid,
            "query": ex["query"],
            "question": ex["question"],
            "choices": list(ex["choices"]) if ex.get("choices") is not None else None,
            "unit": ex.get("unit"),
            "precision": ex.get("precision"),
            "answer": ex["answer"],
            "question_type": ex["question_type"],
            "answer_type": ex["answer_type"],
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
    missing = [r["pid"] for r in recs if not os.path.isfile(os.path.join(args.out_dir, r["image_file"]))]
    print("\n=== DONE ===")
    print("  testmini records: %d   question_types=%s" % (len(recs), qtypes))
    print("  wrote:  %s" % out_json)
    print("  images: %s" % img_dir)
    print("  images missing on disk: %d  (must be 0)" % len(missing))


if __name__ == "__main__":
    main()