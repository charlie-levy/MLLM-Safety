#!/usr/bin/env python3
"""
make_noise_strip.py — Build a labeled horizontal strip of one image with Gaussian
noise at 0%, 10%, ..., 100% (percentage-based, see noise_utils.py).

For presentation visuals: shows what each noise level looks like.

Usage:
  python3 code/make_noise_strip.py <image_path> <out_path> [label]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image, ImageDraw, ImageFont
from noise_utils import noisy_image

LEVELS = list(range(0, 101, 10))  # 0,10,...,100

def make_strip(img_path, out_path, label=""):
    src = Image.open(img_path).convert("RGB")
    thumb = src.copy()
    thumb.thumbnail((220, 220))           # uniform tile size
    tw, th = thumb.size
    pad, top = 8, 30
    n = len(LEVELS)
    strip = Image.new("RGB", (n * (tw + pad) + pad, th + top + pad), "white")
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, pct in enumerate(LEVELS):
        tile = noisy_image(thumb, pct, seed=42)   # fixed seed = reproducible strip
        x = pad + i * (tw + pad)
        strip.paste(tile, (x, top))
        cap = f"{pct}%"
        bb = draw.textbbox((0, 0), cap, font=font)
        draw.text((x + (tw - (bb[2]-bb[0]))//2, 6), cap, fill="black", font=font)

    if label:
        draw.text((pad, th + top - 2), label, fill="#444", font=font)
    strip.save(out_path)
    print("Saved:", out_path, strip.size)

if __name__ == "__main__":
    img = sys.argv[1]
    out = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    make_strip(img, out, label)
