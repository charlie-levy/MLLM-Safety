#!/usr/bin/env python3
"""
make_blur_strip.py — Build a labeled horizontal strip of one image with Gaussian
blur at 0%, 20%, 40%, 60%, 80% (percentage-based, see blur_utils.py).

For presentation visuals: shows what each blur level looks like. Levels match the
bar-chart columns (we stop at 80%, not 100%).

Usage:
  python3 code/make_blur_strip.py <image_path> <out_path> [label]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image, ImageDraw, ImageFont
from blur_utils import blur_image

LEVELS = [0, 20, 40, 60, 80]   # match the bar-chart columns (up to 80%)

def _load_font(size=18):
    for cand in ("DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()

def make_strip_from_image(src, out_path, label=""):
    """Build a 0..80% blur strip from an already-loaded PIL image."""
    src = src.convert("RGB")
    thumb = src.copy()
    thumb.thumbnail((220, 220))
    tw, th = thumb.size
    pad, top = 8, 30
    bottom = 30 if label else pad
    n = len(LEVELS)
    strip = Image.new("RGB", (n * (tw + pad) + pad, th + top + bottom), "white")
    draw = ImageDraw.Draw(strip)
    font = _load_font(18)

    for i, pct in enumerate(LEVELS):
        tile = blur_image(thumb, pct)
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
    make_strip_from_image(Image.open(img_path), out_path, label)

if __name__ == "__main__":
    img = sys.argv[1]
    out = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    make_strip(img, out, label)
