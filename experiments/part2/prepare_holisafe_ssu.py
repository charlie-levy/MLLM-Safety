#!/usr/bin/env python
"""
prepare_holisafe_ssu.py — OFFLINE materializer for the HoliSafe SI+ST->U subset
(type=="SSU": Safe image + Safe text whose correct label is Unsafe — the
compositional/emergent-harm case), the counterpart to the UI+ST (type=="USU")
subset we already built.

Reads the HoliSafe parquet ALREADY IN THE HF CACHE (from the earlier USU
materialization) directly off disk — NO network, NO HfApi, NO hf_hub_download.
The gated online path hangs on this login node; the parquet is already local, so
we glob the cache snapshot and read it. Single-threaded (login-node RLIMIT_NPROC).

Writes the SAME uniform manifest schema as prepare_datasets.py so the Part-10
run_inference.py and the is_refusal judge consume it unchanged:
    /home/ch169788/experiments/part2/data/holisafe_ssu/images/<idx>.png
    /home/ch169788/experiments/part2/data/holisafe_ssu/manifest.jsonl
        {idx, dataset, category, prompt, image_path}

  python prepare_holisafe_ssu.py --inspect_only     # just print type/category counts
  python prepare_holisafe_ssu.py                     # materialize SSU, balanced 500
  python prepare_holisafe_ssu.py --type SSS --name holisafe_sss   # optional: over-refusal control
"""
import os
import io
import json
import glob
import random
import argparse
from collections import Counter

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pyarrow                                     # noqa: E402
pyarrow.set_cpu_count(1)
pyarrow.set_io_thread_count(1)
import pyarrow.parquet as pq                       # noqa: E402
from PIL import Image                              # noqa: E402

OUT_ROOT = "/home/ch169788/experiments/part2/data"
CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--etri-vilab--holisafe-bench")
SEED = 42


def cached_parquets():
    files = sorted(glob.glob(CACHE + "/snapshots/*/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit("no cached HoliSafe parquet under %s — the gated files "
                         "aren't in the HF cache; nothing to read offline." % CACHE)
    return files


def to_pil(val):
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, dict) and val.get("bytes"):
        return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
    if isinstance(val, (bytes, bytearray)):
        return Image.open(io.BytesIO(val)).convert("RGB")
    raise ValueError("cannot decode image (%r)" % (type(val),))


def balanced_pick(items, target, seed=SEED):
    """items: list of (payload, category) -> list of payloads, balanced by category."""
    rng = random.Random(seed)
    by = {}
    for payload, cat in items:
        by.setdefault(str(cat), []).append(payload)
    cats = sorted(by)
    print("  categories (%d):" % len(cats))
    for c in cats:
        print("    %-40s %d" % (c, len(by[c])))
    per = target // max(1, len(cats))
    picked = []
    for c in cats:
        pool = by[c][:]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    rem = target - len(picked)
    if rem > 0:
        left = []
        for c in sorted(cats, key=lambda k: -len(by[k])):
            left.extend(by[c][per:])
        rng.shuffle(left)
        picked.extend(left[:rem])
    rng.shuffle(picked)
    print("  picked %d total" % len(picked))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="SSU", help="HoliSafe type code to keep (default SSU = SI+ST->U)")
    ap.add_argument("--name", default="holisafe_ssu", help="output dataset dir name")
    ap.add_argument("--target", type=int, default=500, help="balanced sample target")
    ap.add_argument("--inspect_only", action="store_true")
    args = ap.parse_args()

    want_type = args.type.upper()
    files = cached_parquets()
    print(">>> HoliSafe %s (offline; %d cached parquet files)" % (want_type, len(files)))

    if args.inspect_only:
        tc, cc = Counter(), Counter()
        for f in files:
            t = pq.read_table(f, columns=["type", "category"])
            typs = [str(x) for x in t.column("type").to_pylist()]
            cats = [str(x) for x in t.column("category").to_pylist()]
            tc.update(typs)
            for ty, ca in zip(typs, cats):
                if ty.upper() == want_type:
                    cc.update([ca])
        print("  all type counts:", dict(sorted(tc.items())))
        print("  %s per-category:" % want_type, dict(sorted(cc.items())))
        return

    # pass 1: collect (file, row_idx_within_file, category) for the wanted type
    items = []
    for f in files:
        t = pq.read_table(f, columns=["type", "category"])
        typs = [str(x) for x in t.column("type").to_pylist()]
        cats = [str(x) for x in t.column("category").to_pylist()]
        for i in range(len(typs)):
            if typs[i].upper() == want_type:
                items.append(((f, i), cats[i]))
    print("  %s rows found: %d" % (want_type, len(items)))
    sel = balanced_pick(items, args.target)
    by_file = {}
    for f, i in sel:
        by_file.setdefault(f, set()).add(i)

    # pass 2: read each file ONE row-group at a time, write only selected rows
    out_dir = os.path.join(OUT_ROOT, args.name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    cols = ["image", "query", "category"]
    idx = 0
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as mf:
        for f, want in by_file.items():
            pf = pq.ParquetFile(f)
            gi = 0
            for rg in range(pf.num_row_groups):
                tb = pf.read_row_group(rg, columns=cols)
                ic, qc, cc = tb.column("image"), tb.column("query"), tb.column("category")
                for j in range(tb.num_rows):
                    if gi in want:
                        ip = os.path.join(img_dir, "%05d.png" % idx)
                        to_pil(ic[j].as_py()).save(ip)
                        mf.write(json.dumps({
                            "idx": idx, "dataset": args.name,
                            "category": str(cc[j].as_py()),
                            "prompt": qc[j].as_py(), "image_path": ip},
                            ensure_ascii=False) + "\n")
                        idx += 1
                    gi += 1
                del tb
    print("[%s] materialized %d samples -> %s/manifest.jsonl" % (args.name, idx, out_dir))


if __name__ == "__main__":
    main()
