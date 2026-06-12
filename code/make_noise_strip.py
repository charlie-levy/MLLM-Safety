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

LEVELS = [0, 20, 40, 60, 80]   # match the bar-chart columns (up to 80%)

def make_strip_from_image(src, out_path, label=""):
    """Build a 0..100% noise strip from an already-loaded PIL image."""
    src = src.convert("RGB")
    thumb = src.copy()
    thumb.thumbnail((220, 220))           # uniform tile size
    tw, th = thumb.size
    pad, top = 8, 30
    bottom = 30 if label else pad          # room for the caption row
    n = len(LEVELS)
    strip = Image.new("RGB", (n * (tw + pad) + pad, th + top + bottom), "white")
    draw = ImageDraw.Draw(strip)
    font = None
    for cand in ("DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            font = ImageFont.truetype(cand, 18); break
        except Exception:
            continue
    if font is None:
        try:  # last resort: borrow matplotlib's bundled DejaVu Sans
            from matplotlib import font_manager
            font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 18)
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
        draw.text((pad, th + top + 6), label, fill="#444", font=font)
    strip.save(out_path)
    print("Saved:", out_path, strip.size)

def make_strip(img_path, out_path, label=""):
    """Build a 0..100% noise strip from an image file path."""
    make_strip_from_image(Image.open(img_path), out_path, label)

if __name__ == "__main__":
    img = sys.argv[1]
    out = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    make_strip(img, out, label)
