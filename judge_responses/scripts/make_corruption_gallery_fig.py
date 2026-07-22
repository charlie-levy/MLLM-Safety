#!/usr/bin/env python3
"""Corruption-gallery figure: one benchmark image (SIUO 9014, the bedroom from
the flip figure) shown clean and through the ten ImageNet-C
corruptions the paper sweeps, at the exact severities used (three blurs at 5,
the rest at 3, custom JPEG). Ties the qualitative flip example to the corruption
bank so a reviewer can see the perceptual range at a glance.
-> REU_WACV/figures/corruption_gallery.{pdf,png}"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.normpath(os.path.join(REPO, ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
import corruption_lib as C  # noqa: E402

OUT = os.path.join(REPO, "REU_WACV", "figures")
IMG = os.path.join(REPO, "example_images", "flip_9014_clean.png")
INK = "#1A1A1A"
plt.rcParams.update({"font.size": 8, "text.color": INK, "pdf.fonttype": 42})

# clean + the ten Part-1 corruptions, in perceptual order
PANELS = [("clean", None)] + [(c, C.severity_for(c)) for c in C.PART1_CORRUPTIONS]
PRETTY = {"elastic_transform": "elastic", "jpeg_compression": "jpeg",
          "defocus_blur": "defocus", "glass_blur": "glass", "motion_blur": "motion",
          "zoom_blur": "zoom", "contrast": "contrast", "frost": "frost",
          "snow": "snow", "fog": "fog"}


def main():
    base = Image.open(IMG).convert("RGB")
    n = len(PANELS)  # 11
    ncol, nrow = 6, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.55))
    for ax in axes.ravel():
        ax.axis("off")
    for k, (name, sev) in enumerate(PANELS):
        ax = axes.ravel()[k]
        img = base if name == "clean" else C.apply_corruption(base, name, severity=sev)
        ax.imshow(img)
        lab = "clean" if name == "clean" else f"{PRETTY[name]} (s{sev})"
        ax.set_title(lab, fontsize=7.6, fontweight="bold" if name == "clean" else "normal",
                     color=INK, pad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.005,
                        wspace=0.06, hspace=0.22)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "corruption_gallery." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else None); print("saved", p)


if __name__ == "__main__":
    main()
