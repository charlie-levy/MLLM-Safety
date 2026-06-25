#!/usr/bin/env python
"""
prepare_datasets.py — ONLINE, ONE-TIME materializer for the Part 2 datasets.

Reads the HF auto-converted parquet branch (refs/convert/parquet) DIRECTLY with
pyarrow — NOT load_dataset. load_dataset's split-generation spawns a CPU-sized
thread pool that core-dumps the login node (RLIMIT_NPROC=100). hf_hub_download is
single-file/single-threaded and pyarrow is capped to 1 thread, so this is safe.

Writes a uniform manifest per dataset:
    /home/ch169788/experiments/part2/data/<dataset>/images/<idx>.png
    /home/ch169788/experiments/part2/data/<dataset>/manifest.jsonl
        {idx, dataset, category, prompt, image_path}

Schemas (confirmed via HF datasets-server, 2026-06):
  mmsafety_tiny : TinyVersion_ID_List.json = LIST of {Scenario, Sampled_ID_List}.
                  config = Scenario minus "NN-" prefix; parquet <config>/SD_TYPO/*;
                  keep rows whose `id` in Sampled_ID_List. prompt=question.
  spa_vl        : parquet test/harm/*  -> prompt=question, image=image, cat=class1. ALL.
  vls_bench     : parquet default/train/*  (7 files, ~3.4GB) -> prompt=instruction,
                  image=image, cat=category. Balanced 500.
  holisafe      : GATED. all parquet rows -> filter type=="USU"; prompt=query,
                  image=image, cat=category. Balanced 500.

  python prepare_datasets.py --dataset all --inspect_only
  python prepare_datasets.py --dataset spa_vl
  python prepare_datasets.py --dataset all
"""
import os
import io
import json
import random
import argparse
from collections import Counter

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import pyarrow                                    # noqa: E402
pyarrow.set_cpu_count(1)
pyarrow.set_io_thread_count(1)
import pyarrow.parquet as pq                      # noqa: E402

import requests                                   # noqa: E402
from PIL import Image                             # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

OUT_ROOT = "/home/ch169788/experiments/part2/data"
SEED = 42
PARQUET_REV = "refs/convert/parquet"
TINY_ID_URL = "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/TinyVersion_ID_List.json"
MM_REPO = "PKU-Alignment/MM-SafetyBench"


# ── parquet helpers ─────────────────────────────────────────────────────────────
def list_parquets(repo, keep):
    """parquet files on the auto-parquet branch where keep(path) is True."""
    fs = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=PARQUET_REV)
                if f.endswith(".parquet") and keep(f))
    if not fs:
        raise SystemExit("%s: no parquet matched on %s" % (repo, PARQUET_REV))
    return fs


def read_table(repo, fname, columns=None):
    local = hf_hub_download(repo, fname, repo_type="dataset", revision=PARQUET_REV)
    return pq.read_table(local, columns=columns)


def to_pil(val):
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, dict) and val.get("bytes"):
        return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
    if isinstance(val, (bytes, bytearray)):
        return Image.open(io.BytesIO(val)).convert("RGB")
    if isinstance(val, str) and os.path.exists(val):
        return Image.open(val).convert("RGB")
    raise ValueError("cannot decode image (%r)" % (type(val),))


def save_rows(name, rows):
    """rows: list of {prompt, image, category}; image is a parquet cell (dict/bytes)."""
    out_dir = os.path.join(OUT_ROOT, name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest, "w", encoding="utf-8") as mf:
        for i, r in enumerate(rows):
            ip = os.path.join(img_dir, "%05d.png" % i)
            to_pil(r["image"]).save(ip)
            mf.write(json.dumps({
                "idx": i, "dataset": name,
                "category": str(r.get("category", "")),
                "prompt": r["prompt"], "image_path": ip,
            }, ensure_ascii=False) + "\n")
    print("[%s] materialized %d samples -> %s" % (name, len(rows), manifest))


def balanced_pick(items, target, seed=SEED):
    """items: list of (payload, category). Return list of payloads, balanced by cat."""
    rng = random.Random(seed)
    by = {}
    for payload, cat in items:
        by.setdefault(str(cat), []).append(payload)
    cats = sorted(by)
    print("  categories (%d):" % len(cats))
    for c in cats:
        print("    %-36s %d" % (c, len(by[c])))
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


def _as_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def stream_write(repo, name, by_file, text_col):
    """Memory-frugal materialize: read each parquet ONE row-group at a time and
    write only the selected rows' images immediately (never load the whole file
    or hold all images in RAM). by_file: {parquet_path: set(row_idx_within_file)}.
    """
    out_dir = os.path.join(OUT_ROOT, name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    cols = ["image", text_col, "category"]
    idx = 0
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as mf:
        for f, want in by_file.items():
            local = hf_hub_download(repo, f, repo_type="dataset", revision=PARQUET_REV)
            pf = pq.ParquetFile(local)
            gi = 0
            for rg in range(pf.num_row_groups):
                tb = pf.read_row_group(rg, columns=cols)
                ic, tc, cc = tb.column("image"), tb.column(text_col), tb.column("category")
                for j in range(tb.num_rows):
                    if gi in want:
                        ip = os.path.join(img_dir, "%05d.png" % idx)
                        to_pil(ic[j].as_py()).save(ip)
                        mf.write(json.dumps({
                            "idx": idx, "dataset": name, "category": str(cc[j].as_py()),
                            "prompt": tc[j].as_py(), "image_path": ip}, ensure_ascii=False) + "\n")
                        idx += 1
                    gi += 1
                del tb
    print("[%s] materialized %d samples -> %s/manifest.jsonl" % (name, idx, out_dir))


def show_schema(repo, fname):
    local = hf_hub_download(repo, fname, repo_type="dataset", revision=PARQUET_REV)
    pf = pq.ParquetFile(local)
    print("  file:", fname)
    print("  columns:", [f.name for f in pf.schema_arrow])
    print("  rows in file:", pf.metadata.num_rows)


# ── per-dataset builders ────────────────────────────────────────────────────────
def do_mmsafety_tiny(inspect_only):
    print("\n>>> MM-SafetyBench-Tiny")
    id_list = requests.get(TINY_ID_URL, timeout=60).json()
    cfg_of = lambda s: s.split("-", 1)[1] if "-" in s else s
    if inspect_only:
        scen = id_list[0]["Scenario"]
        files = list_parquets(MM_REPO, lambda f, c=cfg_of(scen): c in f and "SD_TYPO" in f)
        print(" %s -> %s" % (scen, files))
        show_schema(MM_REPO, files[0])
        return
    rows = []
    for entry in id_list:
        scen = entry["Scenario"]
        cfg = cfg_of(scen)
        want = set(_as_int(x) for x in entry["Sampled_ID_List"])
        files = list_parquets(MM_REPO, lambda f, c=cfg: c in f and "SD_TYPO" in f)
        kept = 0
        for f in files:
            t = read_table(MM_REPO, f)
            ids, qs, imgs = t.column("id").to_pylist(), t.column("question").to_pylist(), t.column("image")
            for i, _id in enumerate(ids):
                if _as_int(_id) in want:
                    rows.append({"prompt": qs[i], "image": imgs[i].as_py(), "category": scen})
                    kept += 1
        print("  %-24s kept %d/%d" % (scen, kept, len(want)))
    save_rows("mmsafety_tiny", rows)


def do_spa_vl(inspect_only):
    print("\n>>> SPA-VL (test/harm)")
    files = list_parquets("sqrti/SPA-VL", lambda f: "harm" in f and "help" not in f)
    print("  files:", files)
    if inspect_only:
        show_schema("sqrti/SPA-VL", files[0])
        return
    rows = []
    for f in files:
        t = read_table("sqrti/SPA-VL", f)
        q, c1, img = t.column("question").to_pylist(), t.column("class1").to_pylist(), t.column("image")
        for i in range(len(q)):
            rows.append({"prompt": q[i], "image": img[i].as_py(), "category": c1[i]})
    save_rows("spa_vl", rows)


def do_vls_bench(inspect_only):
    print("\n>>> VLS-Bench (default/train, balanced 500)")
    files = list_parquets("Foreshhh/vlsbench", lambda f: "train" in f)
    print("  files:", files)
    if inspect_only:
        show_schema("Foreshhh/vlsbench", files[0])
        return
    # pass 1: cheap category read -> (file, row) per category
    items = []
    for f in files:
        cats = read_table("Foreshhh/vlsbench", f, columns=["category"]).column("category").to_pylist()
        items.extend(((f, i), cats[i]) for i in range(len(cats)))
    sel = balanced_pick(items, target=500)            # list of (file, row)
    by_file = {}
    for f, i in sel:
        by_file.setdefault(f, set()).add(i)
    stream_write("Foreshhh/vlsbench", "vls_bench", by_file, "instruction")


def do_holisafe(inspect_only):
    print("\n>>> HoliSafe (gated; type==USU, balanced 500)")
    files = list_parquets("etri-vilab/holisafe-bench", lambda f: True)   # all parquet rows
    print("  files:", files)
    if inspect_only:
        show_schema("etri-vilab/holisafe-bench", files[0])
        return
    # pass 1: category + type (cheap)
    items = []
    for f in files:
        t = read_table("etri-vilab/holisafe-bench", f, columns=["category", "type"])
        cats, typs = t.column("category").to_pylist(), t.column("type").to_pylist()
        for i in range(len(cats)):
            if str(typs[i]).upper() == "USU":
                items.append(((f, i), cats[i]))
    print("  USU rows:", len(items))
    sel = balanced_pick(items, target=500)
    by_file = {}
    for f, i in sel:
        by_file.setdefault(f, set()).add(i)
    stream_write("etri-vilab/holisafe-bench", "holisafe", by_file, "query")


BUILDERS = {
    "mmsafety_tiny": do_mmsafety_tiny,
    "spa_vl": do_spa_vl,
    "vls_bench": do_vls_bench,
    "holisafe": do_holisafe,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all"] + list(BUILDERS))
    ap.add_argument("--inspect_only", action="store_true")
    args = ap.parse_args()
    for name in (list(BUILDERS) if args.dataset == "all" else [args.dataset]):
        try:
            BUILDERS[name](args.inspect_only)
        except Exception as e:
            import traceback
            print("\n[!!] %s failed: %s" % (name, e))
            traceback.print_exc()
            if not args.inspect_only:
                raise


if __name__ == "__main__":
    main()
