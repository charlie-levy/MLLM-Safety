#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preview_blur.py — Phase 1: visual blur sanity check (NO model, NO GPU).

Loads ONE image and saves 20% and 40% Gaussian-blur previews so you can eyeball
the blur intensity before any inference. Uses the SAME blur_image() the eval uses.

  python unsafe_6_19/preview_blur.py
  python unsafe_6_19/preview_blur.py --image unsafe_6_19/Unsafe_Images_BeaverTails-V/image-5.jpg

Writes:
  unsafe_6_19/preview_blur_20pct.png
  unsafe_6_19/preview_blur_40pct.png
  unsafe_6_19/preview_blur_clean.png   (for side-by-side comparison)
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)

from PIL import Image       # noqa: E402
from blur_utils import blur_image   # same blur as the eval  # noqa: E402

DEFAULT_IMG = "unsafe_6_19/Unsafe_Images_BeaverTails-V/image-1.jpg"
OUT = "unsafe_6_19"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMG, help="image to preview")
    args = ap.parse_args()

    assert os.path.exists(args.image), "image not found: %s" % args.image
    im = Image.open(args.image).convert("RGB")
    print("preview source: %s  (size %s)" % (args.image, im.size))

    im.save(os.path.join(OUT, "preview_blur_clean.png"))
    blur_image(im, 20).save(os.path.join(OUT, "preview_blur_20pct.png"))
    blur_image(im, 40).save(os.path.join(OUT, "preview_blur_40pct.png"))

    print("wrote:")
    for f in ("preview_blur_clean.png", "preview_blur_20pct.png", "preview_blur_40pct.png"):
        print("  %s/%s" % (OUT, f))
    print("\nOpen all three and confirm: 20% = mildly blurred, 40% = notably blurred.")


if __name__ == "__main__":
    main()
