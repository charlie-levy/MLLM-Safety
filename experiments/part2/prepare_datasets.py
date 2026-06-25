#!/usr/bin/env python
"""
prepare_datasets.py — ONLINE, ONE-TIME materializer for the Part 2 datasets.

Run on a node WITH internet (login node is fine; single-threaded, saves to disk).
Downloads each dataset, applies the spec's filtering / balanced sampling (seed=42),
saves every image to disk, and writes a uniform manifest:

    /home/ch169788/experiments/part2/data/<dataset>/images/<idx>.png
    /home/ch169788/experiments/part2/data/<dataset>/manifest.jsonl
        {idx, dataset, category, prompt, image_path}

The Slurm inference jobs then read the manifest OFFLINE.

Schemas are now KNOWN (confirmed via HF datasets-server, 2026-06):

  mmsafety_tiny : TinyVersion_ID_List.json is a LIST of {Scenario, Sampled_ID_List}.
                  Per scenario: config = Scenario with the "NN-" prefix stripped
                  (e.g. "07-Sex" -> "Sex"), split SD_TYPO, keep rows whose `id` is
                  in Sampled_ID_List. prompt=question, image=image, category=Scenario.
  spa_vl        : load_dataset("sqrti/SPA-VL","test",split="harm").
                  prompt=question, image=image, category=class1. Use ALL harm rows.
  vls_bench     : load_dataset("Foreshhh/vlsbench",split="train").
                  prompt=instruction, image=image, category=category. Balanced 500.
  holisafe      : GATED (needs HF login + accepted terms). filter type=="USU",
                  prompt=query, image=image, category=category. Balanced 500.

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


# ── helpers ────────────────────────────────────────────────────────────────────
def image_columns(tbl):
    return [name for name, feat in tbl.features.items() if isinstance(feat, datasets.Image)]


def to_pil(val, base_dir=""):
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, dict) and val.get("bytes"):
        return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
    if isinstance(val, str):
        for p in (val, os.path.join(base_dir, val)):
            if p and os.path.exists(p):
                return Image.open(p).convert("RGB")
    raise ValueError("cannot turn value into image (%r)" % (type(val),))


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


def materialize(name, rows, prompt_field, image_field, category_field, base_dir=""):
    out_dir = os.path.join(OUT_ROOT, name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.jsonl")
    n = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        for i, ex in enumerate(rows):
            img = to_pil(ex[image_field], base_dir)
            ip = os.path.join(img_dir, "%05d.png" % i)
            img.save(ip)
            mf.write(json.dumps({
                "idx": i,
                "dataset": name,
                "category": str(ex.get(category_field, "")) if category_field else "",
                "prompt": ex[prompt_field],
                "image_path": ip,
            }, ensure_ascii=False) + "\n")
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
        print("    %-34s %d" % (c, len(by_cat[c])))
    per = target // max(1, len(cats))
    picked = []
    for c in cats:
        pool = by_cat[c][:]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    remainder = target - len(picked)
    if remainder > 0:
        leftovers = []
        for c in sorted(cats, key=lambda k: -len(by_cat[k])):
            leftovers.extend(by_cat[c][per:])
        rng.shuffle(leftovers)
        picked.extend(leftovers[:remainder])
    rng.shuffle(picked)
    print("  final per-category:", dict(Counter(str(ex.get(category_field, "")) for ex in picked)))
    print("  total picked:", len(picked))
    return picked


def _as_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


# ── per-dataset builders ────────────────────────────────────────────────────────
def do_mmsafety_tiny(inspect_only):
    print("\n>>> MM-SafetyBench-Tiny: fetching Tiny ID list")
    id_list = requests.get(TINY_ID_URL, timeout=60).json()   # LIST of {Scenario, Sampled_ID_List}
    print("  %d scenarios; e.g. %s -> %d ids"
          % (len(id_list), id_list[0]["Scenario"], len(id_list[0]["Sampled_ID_List"])))

    def cfg_of(scenario):
        return scenario.split("-", 1)[1] if "-" in scenario else scenario

    if inspect_only:
        scen = id_list[0]["Scenario"]
        ds = load_dataset("PKU-Alignment/MM-SafetyBench", name=cfg_of(scen))
        inspect("mmsafety_tiny[%s]" % cfg_of(scen), ds)
        return

    rows = []
    for entry in id_list:
        scen = entry["Scenario"]
        want = set(_as_int(x) for x in entry["Sampled_ID_List"])
        ds = load_dataset("PKU-Alignment/MM-SafetyBench", name=cfg_of(scen))
        split = "SD_TYPO" if "SD_TYPO" in ds else list(ds.keys())[0]
        tbl = ds[split]
        kept = 0
        for ex in tbl:
            if _as_int(ex.get("id")) in want:
                ex = dict(ex)
                ex["_category"] = scen
                rows.append(ex)
                kept += 1
        print("  %-24s kept %d/%d (split=%s)" % (scen, kept, len(want), split))
    if not rows:
        raise SystemExit("mmsafety_tiny: no rows matched Tiny ids — check the `id` field.")
    materialize("mmsafety_tiny", rows, "question", "image", "_category")


def do_spa_vl(inspect_only):
    ds = load_dataset("sqrti/SPA-VL", "test", split="harm")   # config=test, split=harm
    inspect("spa_vl", ds)
    if inspect_only:
        return
    materialize("spa_vl", list(ds), "question", "image", "class1")


def do_vls_bench(inspect_only):
    ds = load_dataset("Foreshhh/vlsbench", split="train")
    inspect("vls_bench", ds)
    if inspect_only:
        return
    picked = balanced_subset(list(ds), "category", target=500)
    materialize("vls_bench", picked, "instruction", "image", "category")


def do_holisafe(inspect_only):
    # GATED: needs `huggingface-cli login` + accepted terms on the dataset page.
    ds = load_dataset("etri-vilab/holisafe-bench")
    first = inspect("holisafe", ds)
    if inspect_only:
        return
    split = list(ds.keys())[0] if isinstance(ds, datasets.DatasetDict) else None
    tbl = ds[split] if split else ds
    rows = [ex for ex in tbl if str(ex.get("type", "")).upper() == "USU"]
    print("  filtered type==USU: %d/%d" % (len(rows), len(tbl)))
    if not rows:
        raise SystemExit("holisafe: no USU rows — check the `type` field values.")
    picked = balanced_subset(rows, "category", target=500)
    # image may be an Image feature OR a relative path string; handle both.
    img_field = (image_columns(tbl) or ["image"])[0]
    base = os.path.dirname(getattr(tbl, "cache_files", [{}])[0].get("filename", "")) if tbl.cache_files else ""
    materialize("holisafe", picked, "query", img_field, "category", base_dir=base)


BUILDERS = {
    "mmsafety_tiny": do_mmsafety_tiny,
    "spa_vl": do_spa_vl,
    "vls_bench": do_vls_bench,
    "holisafe": do_holisafe,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all"] + list(BUILDERS))
    ap.add_argument("--inspect_only", action="store_true",
                    help="print structure (splits/cols/first/category counts) and EXIT")
    args = ap.parse_args()
    for name in (list(BUILDERS) if args.dataset == "all" else [args.dataset]):
        try:
            BUILDERS[name](args.inspect_only)
        except Exception as e:
            print("\n[!!] %s failed: %s" % (name, e))
            if not args.inspect_only:
                raise


if __name__ == "__main__":
    main()
