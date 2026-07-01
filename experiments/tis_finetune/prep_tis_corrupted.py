#!/usr/bin/env python
"""
prep_tis_corrupted.py — build a CORRUPTED COPY of the Think-in-Safety dataset for
the apples-to-apples "TIS + random image corruptions" LoRA run.

Run on Newton in the REU env (has imagecorruptions / corruption_lib). NO GPU and
NO internet needed — it only reads the already-prepped clean images and writes new ones.

Each training image is randomly assigned ONE condition (uniform over 4):
    clean | zoom_blur (sev 3) | snow (sev 3) | glass_blur (sev 5)
Corruptions use the SAME corruption_lib.apply_corruption + severity_for as every past
experiment (Parts 1/3/4) — same library, same severities, same call.

Writes a FULL SELF-CONTAINED COPY so the original 'think_in_safety' dataset is UNTOUCHED:
    clean-assigned images  -> copied verbatim
    corrupted images       -> corrupted PNG
into  <LF>/data/think_in_safety_corrupt/  plus  think_in_safety_corrupt.json,
registers dataset 'think_in_safety_corrupt', and writes a provenance manifest CSV.

  conda activate REU
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  python3 prep_tis_corrupted.py            # ~30-60 min (glass_blur is CPU-slow)
"""
import os
import sys
import json
import csv
import random
import shutil
import argparse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                 # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))
import numpy as np                                            # noqa: E402
from corruption_lib import apply_corruption, severity_for      # noqa: E402

CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]      # uniform random assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lf_dir", default="~/LLaMA-Factory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lf = os.path.abspath(os.path.expanduser(args.lf_dir))
    data_dir = os.path.join(lf, "data")
    clean_base = os.path.join(data_dir, "think_in_safety")
    corrupt_base = os.path.join(data_dir, "think_in_safety_corrupt")
    clean_json = os.path.join(data_dir, "think_in_safety.json")
    if not os.path.isfile(clean_json):
        raise SystemExit("clean dataset not found: %s (run prep_tis_data.py first)" % clean_json)
    os.makedirs(corrupt_base, exist_ok=True)

    recs = json.load(open(clean_json))
    rng = random.Random(args.seed)          # condition assignment stream
    np.random.seed(args.seed)               # imagecorruptions' internal randomness (as in past runs)

    counts = {c: 0 for c in CONDITIONS}
    manifest = []
    missing = n_img = 0
    for ri, r in enumerate(recs):
        new_imgs = []
        for ip in (r.get("images", []) or []):
            n_img += 1
            if not os.path.isfile(ip):
                missing += 1
            rel = os.path.relpath(ip, clean_base)               # e.g. bad_ads/uuid.png
            dst = os.path.join(corrupt_base, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            cond = rng.choice(CONDITIONS)
            counts[cond] += 1
            if cond == "clean":
                shutil.copyfile(ip, dst)                        # verbatim copy
            else:
                out = apply_corruption(Image.open(ip).convert("RGB"),
                                       cond, severity=severity_for(cond))
                out.save(dst)
            new_imgs.append(dst)
            manifest.append((rel, cond, 0 if cond == "clean" else severity_for(cond)))
        r["images"] = new_imgs
        if (ri + 1) % 200 == 0:
            print("  %d/%d records | counts=%s" % (ri + 1, len(recs), counts), flush=True)

    # corrupted dataset json
    out_json = os.path.join(data_dir, "think_in_safety_corrupt.json")
    json.dump(recs, open(out_json, "w"), ensure_ascii=False)

    # register (idempotent)
    di_path = os.path.join(data_dir, "dataset_info.json")
    di = json.load(open(di_path)) if os.path.isfile(di_path) else {}
    di["think_in_safety_corrupt"] = {
        "file_name": "think_in_safety_corrupt.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {"role_tag": "from", "content_tag": "value",
                 "user_tag": "human", "assistant_tag": "gpt"},
    }
    json.dump(di, open(di_path, "w"), ensure_ascii=False, indent=2)

    # provenance manifest (which image got which corruption)
    man_path = os.path.join(data_dir, "think_in_safety_corrupt_manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_rel", "condition", "severity"])
        w.writerows(manifest)

    print("\n=== DONE ===")
    print("records: %d | images: %d | missing source: %d  (must be 0)" % (len(recs), n_img, missing))
    print("condition counts (uniform ~25%% each): %s" % counts)
    print("wrote:      %s" % out_json)
    print("registered: 'think_in_safety_corrupt' in %s" % di_path)
    print("manifest:   %s" % man_path)
    print("\n" + ("READY — corrupted copy registered as 'think_in_safety_corrupt'."
                  if missing == 0 else "WARNINGS: missing source images above."))


if __name__ == "__main__":
    main()
