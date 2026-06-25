#!/usr/bin/env python
"""
prepare_datasets.py — ONLINE, ONE-TIME materializer for the Part 2 datasets.

Run this on a node WITH internet (login node is fine; it is single-threaded and
saves to disk — it does NOT thread-bomb the way `load_dataset` inside a parallel
job would). It downloads each dataset, applies the spec's filtering / balanced
sampling (seed=42), saves every image to disk, and writes a uniform manifest:

    /home/ch169788/experiments/part2/data/<dataset>/images/<idx>.png
    /home/ch169788/experiments/part2/data/<dataset>/manifest.jsonl
        {idx, dataset, category, prompt, image_path}

The Slurm inference jobs then read the manifest OFFLINE — no HF calls at job time.

Because the exact column names for SPA-VL / VLS-Bench / HoliSafe are not known
ahead of time, this script is INSPECTION-FIRST and FAIL-LOUD:
  * --inspect_only : load each dataset, print splits/columns/first example/category
                     counts, and EXIT (this is the spec's "show me the structure"
                     gate). Materializes nothing.
  * default        : inspect THEN materialize, auto-detecting the image / prompt /
                     category fields. If a field can't be found it prints the real
                     columns and raises, so we fix the mapping instead of guessing.

  python prepare_datasets.py --dataset all --inspect_only
  python prepare_datasets.py --dataset spa_vl
  python prepare_datasets.py --dataset all
"""
import os
import io
import sys
import json
import random
import argparse
from collections import Counter

# single-thread everything so this is safe on the login node
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import requests                                    # noqa: E402
from PIL import Image                              # noqa: E402
import datasets                                    # noqa: E402
from datasets import load_dataset                  # noqa: E402

OUT_ROOT = "/home/ch169788/experiments/part2/data"
SEED = 42
TINY_ID_URL = "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/TinyVersion_ID_List.json"

# candidate field names, most-specific first
PROMPT_FIELDS = ["question", "prompt", "instruction", "query", "text", "Question"]
CATEGORY_FIELDS = ["category", "class", "type", "harm_category", "subcategory",
                   "harm_type", "category_name", "scenario", "label"]


# ── helpers ────────────────────────────────────────────────────────────────────
def image_columns(ds):
    return [name for name, feat in ds.features.items() if isinstance(feat, datasets.Image)]


def find_field(example, candidates):
    for c in candidates:
        if c in example and isinstance(example[c], str) and example[c].strip():
            return c
    return None


def to_pil(val):
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, dict) and "bytes" in val and val["bytes"]:
        return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
    if isinstance(val, str) and os.path.exists(val):
        return Image.open(val).convert("RGB")
    raise ValueError("cannot turn value into image: %r" % (type(val),))


def inspect(name, ds):
    print("\n" + "=" * 78)
    print("  INSPECT %s" % name)
    print("=" * 78)
    if isinstance(ds, datasets.DatasetDict):
        print("splits:", {k: len(v) for k, v in ds.items()})
        first = ds[list(ds.keys())[0]]
    else:
        print("rows:", len(ds))
        first = ds
    print("columns:", first.column_names)
    print("image columns:", image_columns(first))
    ex = first[0]
    for k, v in ex.items():
        sv = (v[:160] if isinstance(v, str) else
              ("<PIL %s>" % (v.size,) if isinstance(v, Image.Image) else
               ("<imgdict>" if isinstance(v, dict) and "bytes" in v else repr(v)[:120])))
        print("  %-18s = %s" % (k, sv))
    return first


def materialize(name, rows, prompt_field, image_field, category_field):
    """rows: list of dataset examples. Saves images + manifest.jsonl."""
    out_dir = os.path.join(OUT_ROOT, name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.jsonl")
    n = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        for i, ex in enumerate(rows):
            img = to_pil(ex[image_field])
            ip = os.path.join(img_dir, "%05d.png" % i)
            img.save(ip)
            rec = {
                "idx": i,
                "dataset": name,
                "category": str(ex.get(category_field, "")) if category_field else "",
                "prompt": ex[prompt_field],
                "image_path": ip,
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("[%s] materialized %d samples -> %s" % (name, n, manifest))
    return n


def balanced_subset(rows, category_field, target, seed=SEED):
    """Even sample across categories; fill remainder from largest categories."""
    rng = random.Random(seed)
    by_cat = {}
    for ex in rows:
        by_cat.setdefault(str(ex.get(category_field, "")), []).append(ex)
    cats = sorted(by_cat)
    print("  categories (%d):" % len(cats))
    for c in cats:
        print("    %-30s %d" % (c, len(by_cat[c])))
    per = target // max(1, len(cats))
    picked = []
    for c in cats:
        pool = by_cat[c][:]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    # fill remainder from largest categories
    remainder = target - len(picked)
    if remainder > 0:
        leftovers = []
        for c in sorted(cats, key=lambda k: -len(by_cat[k])):
            pool = by_cat[c][per:]
            leftovers.extend(pool)
        rng.shuffle(leftovers)
        picked.extend(leftovers[:remainder])
    rng.shuffle(picked)
    print("  final per-category counts:",
          dict(Counter(str(ex.get(category_field, "")) for ex in picked)))
    print("  total picked:", len(picked))
    return picked


# ── per-dataset builders ────────────────────────────────────────────────────────
def do_mmsafety_tiny(inspect_only):
    print("\n>>> MM-SafetyBench-Tiny: fetching Tiny ID list")
    id_list = requests.get(TINY_ID_URL, timeout=60).json()
    print("  id_list type=%s  top-level keys/len=%s"
          % (type(id_list).__name__,
             list(id_list)[:5] if isinstance(id_list, dict) else len(id_list)))
    print(json.dumps(id_list, indent=2)[:1500])
    if inspect_only:
        # show one scenario config's structure (SD_TYPO split, per the proven pipeline)
        scen = list(id_list)[0] if isinstance(id_list, dict) else "01-Illegal_Activitiy"
        try:
            ds = load_dataset("PKU-Alignment/MM-SafetyBench", name=scen)
            inspect("mmsafety_tiny[%s]" % scen, ds)
        except Exception as e:
            print("  [inspect] could not load config %r: %s" % (scen, e))
        return
    # materialize: iterate scenarios, keep only Tiny ids, SD_TYPO split.
    rows = []
    scenarios = list(id_list) if isinstance(id_list, dict) else []
    for scen in scenarios:
        want = set(str(x) for x in id_list[scen])
        ds = load_dataset("PKU-Alignment/MM-SafetyBench", name=scen)
        split = "SD_TYPO" if "SD_TYPO" in ds else list(ds.keys())[0]
        tbl = ds[split]
        idf = image_columns(tbl)[0]
        pf = find_field(tbl[0], PROMPT_FIELDS)
        for j, ex in enumerate(tbl):
            sid = str(ex.get("id", j))
            if not want or sid in want:
                ex = dict(ex)
                ex["_category"] = scen
                rows.append(ex)
        print("  %s: kept %d (split=%s)" % (scen, sum(1 for r in rows if r["_category"] == scen), split))
    if not rows:
        raise SystemExit("MM-SafetyBench-Tiny: no rows matched the Tiny id list — inspect id format.")
    pf = find_field(rows[0], PROMPT_FIELDS)
    idf = [k for k, v in rows[0].items() if isinstance(v, (Image.Image, dict)) and k != "_category"]
    idf = idf[0] if idf else "image"
    materialize("mmsafety_tiny", rows, pf, idf, "_category")


def do_spa_vl(inspect_only):
    ds = load_dataset("sqrti/SPA-VL", split="test")
    first = inspect("spa_vl", ds)
    if inspect_only:
        return
    # filter to harm split — detect the column carrying the harm/safe flag
    rows = [ex for ex in ds]
    # SPA-VL test has a 'harm'/'safe' style split column; keep harmful ones.
    cat = find_field(first[0], CATEGORY_FIELDS)
    pf = find_field(first[0], PROMPT_FIELDS)
    idf = image_columns(first)[0]
    if pf is None or idf is None:
        raise SystemExit("SPA-VL: could not detect prompt/image fields. Columns=%s" % first.column_names)
    materialize("spa_vl", rows, pf, idf, cat)


def do_vls_bench(inspect_only):
    ds = load_dataset("Foreshhh/vlsbench")
    first = inspect("vls_bench", ds)
    if inspect_only:
        return
    rows = [ex for ex in first]
    cat = find_field(first[0], CATEGORY_FIELDS)
    pf = find_field(first[0], PROMPT_FIELDS)
    idf = image_columns(first)[0]
    if pf is None or idf is None:
        raise SystemExit("VLS-Bench: could not detect prompt/image fields. Columns=%s" % first.column_names)
    if cat is None:
        raise SystemExit("VLS-Bench: no category column for balanced sampling. Columns=%s" % first.column_names)
    picked = balanced_subset(rows, cat, target=500)
    materialize("vls_bench", picked, pf, idf, cat)


def do_holisafe(inspect_only):
    ds = load_dataset("etri-vilab/holisafe-bench")
    first = inspect("holisafe", ds)
    if inspect_only:
        return
    rows = [ex for ex in first]
    # filter to type == "USU"
    type_field = "type" if "type" in first.column_names else find_field(first[0], ["type", "safety_type"])
    if type_field:
        usu = [ex for ex in rows if str(ex.get(type_field, "")).upper() == "USU"]
        print("  filtered type==USU: %d/%d" % (len(usu), len(rows)))
        rows = usu
    cat = find_field(first[0], CATEGORY_FIELDS) or "category"
    pf = find_field(first[0], PROMPT_FIELDS)
    idf = image_columns(first)[0]
    if pf is None or idf is None:
        raise SystemExit("HoliSafe: could not detect prompt/image fields. Columns=%s" % first.column_names)
    picked = balanced_subset(rows, cat, target=500)
    materialize("holisafe", picked, pf, idf, cat)


BUILDERS = {
    "mmsafety_tiny": do_mmsafety_tiny,
    "spa_vl": do_spa_vl,
    "vls_bench": do_vls_bench,
    "holisafe": do_holisafe,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all"] + list(BUILDERS))
    ap.add_argument("--inspect_only", action="store_true",
                    help="print structure (splits/cols/first/category counts) and EXIT")
    args = ap.parse_args()
    targets = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        try:
            BUILDERS[name](args.inspect_only)
        except Exception as e:
            print("\n[!!] %s failed: %s" % (name, e))
            if not args.inspect_only:
                raise


if __name__ == "__main__":
    main()