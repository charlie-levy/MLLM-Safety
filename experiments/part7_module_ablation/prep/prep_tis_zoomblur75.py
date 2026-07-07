#!/usr/bin/env python
"""
prep_tis_zoomblur75.py — build a CORRUPTED COPY of the Think-in-Safety dataset for the
module-ablation study: 75% of images get zoom_blur(severity=2), 25% stay clean.

APPLES-TO-APPLES with prep_tis_corrupted.py (same corruption_lib, same self-contained-copy
pattern, same dataset_info registration, same manifest). The ONLY differences:
  * ONE corruption family: zoom_blur at severity 2 (matches Part 6 SIUO eval sev2 exactly),
  * a 75/25 corrupt/clean split (per image, seeded) instead of a uniform 4-way split.

The original 'think_in_safety' dataset is NEVER modified. Writes:
    <LF>/data/think_in_safety_zoomblur75/              (full self-contained image copy)
    <LF>/data/think_in_safety_zoomblur75.json          (sharegpt records -> new image paths)
    <LF>/data/think_in_safety_zoomblur75_manifest.csv  (which image got which condition)
    <LF>/data/think_in_safety_zoomblur75_PROOF.png     (before/after montage — SEE the blur)
and registers dataset 'think_in_safety_zoomblur75' in dataset_info.json.

PROOF that the corruption is really applied (printed + saved):
  * condition counts (~75% zoom_blur / ~25% clean),
  * mean-abs pixel diff vs the ORIGINAL for a random sample:
        zoom_blur images MUST have MAD >> 0 (pixels changed),
        clean       images MUST have MAD == 0 (byte-identical copy),
  * a side-by-side montage PNG of several [original | zoom_blur(sev2)] pairs.

Run on Newton in the REU env (has imagecorruptions / corruption_lib). NO GPU, NO internet.
  conda activate REU
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  python3 prep_tis_zoomblur75.py                 # ~10-20 min (zoom_blur is CPU-moderate)
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
# .../experiments/part7_module_ablation/prep -> repo root is three dirs up
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))
import numpy as np                                             # noqa: E402
from corruption_lib import apply_corruption                    # noqa: E402

CORRUPTION = "zoom_blur"
SEVERITY = 2                 # EXPLICIT sev2 (matches Part 6 SIUO eval); NOT severity_for()
CORRUPT_FRACTION = 0.75      # 75% of images get zoom_blur(sev2); 25% stay clean
DATASET_NAME = "think_in_safety_zoomblur75"


def mad(a_path, b_img):
    """Mean absolute per-pixel difference between an on-disk image and a PIL image."""
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.int16)
    b = np.asarray(b_img.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return float("nan")
    return float(np.abs(a - b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lf_dir", default="~/LLaMA-Factory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proof_n", type=int, default=6, help="# of pairs in the montage")
    args = ap.parse_args()

    lf = os.path.abspath(os.path.expanduser(args.lf_dir))
    data_dir = os.path.join(lf, "data")
    clean_base = os.path.join(data_dir, "think_in_safety")
    corrupt_base = os.path.join(data_dir, DATASET_NAME)
    clean_json = os.path.join(data_dir, "think_in_safety.json")
    if not os.path.isfile(clean_json):
        raise SystemExit("clean dataset not found: %s (run prep_tis_data.py first)" % clean_json)
    os.makedirs(corrupt_base, exist_ok=True)

    recs = json.load(open(clean_json))
    rng = random.Random(args.seed)          # condition-assignment stream (seeded -> reproducible)
    np.random.seed(args.seed)               # imagecorruptions internal randomness (as in past runs)

    counts = {"clean": 0, CORRUPTION: 0}
    manifest = []
    proof_pairs = []                        # (orig_path, corrupted_dst) for the montage
    mad_zoom, mad_clean = [], []            # MAD samples for numeric proof
    missing = n_img = 0

    for ri, r in enumerate(recs):
        new_imgs = []
        for ip in (r.get("images", []) or []):
            n_img += 1
            if not os.path.isfile(ip):
                missing += 1
                new_imgs.append(ip)
                continue
            rel = os.path.relpath(ip, clean_base)               # e.g. bad_ads/uuid.png
            dst = os.path.join(corrupt_base, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            corrupt_this = rng.random() < CORRUPT_FRACTION
            if not corrupt_this:
                counts["clean"] += 1
                shutil.copyfile(ip, dst)                        # verbatim (byte-identical) copy
                if len(mad_clean) < 50:
                    mad_clean.append(mad(ip, Image.open(dst)))  # must be 0.0
                manifest.append((rel, "clean", 0))
            else:
                counts[CORRUPTION] += 1
                out = apply_corruption(Image.open(ip).convert("RGB"), CORRUPTION, severity=SEVERITY)
                out.save(dst)
                if len(mad_zoom) < 50:
                    mad_zoom.append(mad(ip, out))               # must be >> 0
                if len(proof_pairs) < args.proof_n:
                    proof_pairs.append((ip, dst))
                manifest.append((rel, CORRUPTION, SEVERITY))
            new_imgs.append(dst)
        r["images"] = new_imgs
        if (ri + 1) % 200 == 0:
            print("  %d/%d records | counts=%s" % (ri + 1, len(recs), counts), flush=True)

    # corrupted dataset json (points at the new image copies)
    out_json = os.path.join(data_dir, DATASET_NAME + ".json")
    json.dump(recs, open(out_json, "w"), ensure_ascii=False)

    # register (idempotent) — identical shape to think_in_safety
    di_path = os.path.join(data_dir, "dataset_info.json")
    di = json.load(open(di_path)) if os.path.isfile(di_path) else {}
    di[DATASET_NAME] = {
        "file_name": DATASET_NAME + ".json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {"role_tag": "from", "content_tag": "value",
                 "user_tag": "human", "assistant_tag": "gpt"},
    }
    json.dump(di, open(di_path, "w"), ensure_ascii=False, indent=2)

    # provenance manifest
    man_path = os.path.join(data_dir, DATASET_NAME + "_manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_rel", "condition", "severity"])
        w.writerows(manifest)

    # visual proof montage: rows of [ORIGINAL | zoom_blur(sev2)]
    proof_path = os.path.join(data_dir, DATASET_NAME + "_PROOF.png")
    if proof_pairs:
        thumbs = []
        cellw = 320
        for orig, corr in proof_pairs:
            o = Image.open(orig).convert("RGB")
            c = Image.open(corr).convert("RGB")
            h = int(cellw * o.height / o.width)
            o = o.resize((cellw, h)); c = c.resize((cellw, h))
            row = Image.new("RGB", (cellw * 2 + 8, h), (255, 255, 255))
            row.paste(o, (0, 0)); row.paste(c, (cellw + 8, 0))
            thumbs.append(row)
        W = max(t.width for t in thumbs)
        H = sum(t.height for t in thumbs) + 8 * (len(thumbs) - 1)
        montage = Image.new("RGB", (W, H), (255, 255, 255))
        y = 0
        for t in thumbs:
            montage.paste(t, (0, y)); y += t.height + 8
        montage.save(proof_path)

    # ---- report / PROOF ----
    tot = counts["clean"] + counts[CORRUPTION]
    frac = counts[CORRUPTION] / tot if tot else 0.0
    zmean = sum(mad_zoom) / len(mad_zoom) if mad_zoom else 0.0
    cmean = sum(mad_clean) / len(mad_clean) if mad_clean else 0.0
    print("\n=== DONE ===")
    print("records: %d | images: %d | missing source: %d  (must be 0)" % (len(recs), n_img, missing))
    print("condition counts: %s   -> zoom_blur fraction = %.3f (target %.2f)" % (counts, frac, CORRUPT_FRACTION))
    print("wrote:      %s" % out_json)
    print("registered: '%s' in %s" % (DATASET_NAME, di_path))
    print("manifest:   %s" % man_path)
    print("montage:    %s   (scp this to SEE the blur)" % proof_path)
    print("\n--- PROOF corruption is applied (pixel-level) ---")
    print("  zoom_blur(sev2) mean|Δpixel| vs original (n=%d): %.2f   MUST be >> 0" % (len(mad_zoom), zmean))
    print("  clean          mean|Δpixel| vs original (n=%d): %.2f   MUST be == 0" % (len(mad_clean), cmean))
    ok = (missing == 0 and zmean > 1.0 and cmean == 0.0
          and abs(frac - CORRUPT_FRACTION) < 0.05)
    print("\n" + ("READY ✅ — corruption verified applied; dataset registered as '%s'." % DATASET_NAME
                  if ok else "❌ CHECK ABOVE — proof thresholds not met."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
