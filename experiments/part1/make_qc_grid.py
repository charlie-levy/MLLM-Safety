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

import numpy as np                                             # noqa: E402
from PIL import Image                                          # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

import run_eval as RE                                          # noqa: E402  (chdir's to REPO)
from dataset_loader import load_figstep, load_new_attack       # noqa: E402
from corruption_lib import apply_corruption, PART1_CORRUPTIONS, severity_for  # noqa: E402


def first_image(samples):
    for s in samples:
        if s.get("image") is not None:
            return s["image"].convert("RGB")
    raise RuntimeError("no usable image in dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ch169788/experiments/part1/qc_corruption_grid.png")
    args = ap.parse_args()

    rows = [("FigStep", first_image(load_figstep())),
            ("SIUO", first_image(load_new_attack("siuo")))]

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
                ax.set_title("%s\nsev%d" % (corr, sev), fontsize=7)
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