#!/usr/bin/env python
"""
inspect_datasets.py — Part 2 post-materialization STOP gate.

Reads each materialized manifest under part2/data/<dataset>/ and prints exactly
what the spec asks for before inference: sample count, fields, first 3 prompts,
one image size, and the category histogram. Reads local files only (offline).

  python inspect_datasets.py
"""
import os
import json
import argparse
from collections import Counter
from PIL import Image

DATA_ROOT = "/home/ch169788/experiments/part2/data"
DATASETS = ["mmsafety_tiny", "spa_vl", "vls_bench", "holisafe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    args = ap.parse_args()

    for name in DATASETS:
        mpath = os.path.join(args.data_root, name, "manifest.jsonl")
        print("\n" + "=" * 78)
        print("  %s" % name)
        print("=" * 78)
        if not os.path.exists(mpath):
            print("  [missing] %s — run prepare_datasets.py --dataset %s" % (mpath, name))
            continue
        recs = [json.loads(l) for l in open(mpath) if l.strip()]
        print("  sample count :", len(recs))
        print("  fields       :", list(recs[0].keys()))
        print("  categories   :", dict(Counter(r["category"] for r in recs)))
        print("  first 3 prompts:")
        for r in recs[:3]:
            print("    - [%s] %s" % (r["category"], r["prompt"][:140].replace("\n", " ")))
        ip = recs[0]["image_path"]
        if os.path.exists(ip):
            print("  sample image :", Image.open(ip).size, ip)
        else:
            print("  [warn] first image missing:", ip)


if __name__ == "__main__":
    main()