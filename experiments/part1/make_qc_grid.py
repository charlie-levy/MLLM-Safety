#!/usr/bin/env python
"""
make_qc_grid.py — Part 1 corruption QC gate.

Loads one FigStep image and one SIUO image, applies all 10 Part-1 corruptions at
severity=3, and saves a 2-row x 10-col visual grid (clean is shown as the first
column label only — the spec asks for the 10 corruptions). Prints each corrupted
image's size + dtype to confirm no crashes. STOP gate: eyeball the grid before
any inference.

  python make_qc_grid.py
  python make_qc_grid.py --out /home/ch169788/experiments/part1/qc_corruption_grid.png
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))

import glob                                                    # noqa: E402
import json                                                    # noqa: E402
import numpy as np                                             # noqa: E402
from PIL import Image                                          # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

# NOTE: deliberately does NOT import run_eval/torch or call load_figstep/
# load_new_attack — those open every image in both datasets (500+167) off Lustre
# just to grab one. We open exactly ONE image per dataset, so this is seconds.
import dataset_loader as DL                                    # noqa: E402  (no torch)
import pandas as pd                                            # noqa: E402
from corruption_lib import apply_corruption, PART1_CORRUPTIONS, severity_for, JPEG_QUALITY  # noqa: E402


def one_figstep_image():
    df = pd.read_csv(os.path.join(DL.FIGSTEP_REPO_PATH, DL.FIGSTEP_CSV))
    for _, row in df.iterrows():
        pat = os.path.join(DL.FIGSTEP_REPO_PATH, DL.FIGSTEP_IMAGE_DIR,
                           "query_%s_%s_%s_*.png" % (row["dataset"], row["category_id"], row["task_id"]))
        m = glob.glob(pat)
        if m:
            return Image.open(m[0]).convert("RGB")
    raise RuntimeError("no FigStep image found under %s" % DL.FIGSTEP_IMAGE_DIR)


def one_siuo_image():
    sj = os.path.join(DL.NEW_ATTACKS_DIR, "siuo", "siuo.json")
    data = json.load(open(sj, encoding="utf-8"))
    for item in data.values():
        ip = item.get("image_path", "")
        if ip and os.path.exists(ip):
            return Image.open(ip).convert("RGB")
    raise RuntimeError("no SIUO image found via %s" % sj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ch169788/experiments/part1/qc_corruption_grid.png")
    args = ap.parse_args()

    print("loading one FigStep + one SIUO image ...", flush=True)
    rows = [("FigStep", one_figstep_image()),
            ("SIUO", one_siuo_image())]

    ncol = len(PART1_CORRUPTIONS)
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.4))
    for r, (label, img) in enumerate(rows):
        print("\n[%s] orig size=%s" % (label, img.size))
        for c, corr in enumerate(PART1_CORRUPTIONS):
            sev = severity_for(corr)
            out = apply_corruption(img, corr, severity=sev)
            arr = np.asarray(out)
            print("  %-18s sev=%d size=%s dtype=%s" % (corr, sev, out.size, arr.dtype))
            ax = axes[r][c]
            ax.imshow(out)
            ax.axis("off")
            if r == 0:
                lbl = ("%s\nq%d" % (corr, JPEG_QUALITY)) if corr == "jpeg_compression" \
                    else ("%s\nsev%d" % (corr, sev))
                ax.set_title(lbl, fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    plt.suptitle("Part 1 QC — 10 ImageNet-C corruptions (5 of them @ sev5, rest @ sev3)", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print("\nSaved QC grid -> %s" % args.out)
    print("STOP: eyeball the grid, confirm corruptions are visible but not total wipeout.")


if __name__ == "__main__":
    main()