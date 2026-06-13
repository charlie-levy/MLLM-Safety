#!/usr/bin/env python3
"""
make_corruption_strips.py — Build labeled 0..80% example strips for EVERY corruption
(gaussian noise, gaussian blur, JPEG, brightness/low-light, pixelation) on one real
image from EACH safety dataset (FigStep, XSTest, MMSA), using the SAME dataset loaders
the evals use — so the strips are exactly the images the model is scored on.

This is the "show the inputs" companion to the eval: for every corruption we report,
there is a picture of what that corruption does to a real benchmark image.

Run ON NEWTON from the repo root:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 code/make_corruption_strips.py [fig_idx] [xstest_idx] [mmsa_idx]

Writes results/corruption_examples/<corruption>_strip_<dataset>.png
Pull that folder to your Mac for the slides.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))   # repo root

from PIL import Image, ImageDraw, ImageFont
from dataset_loader import load_figstep, load_xstest, load_mmsa
from noise_utils import noisy_image
from blur_utils import blur_image
from jpeg_utils import jpeg_image
from brightness_utils import brightness_image
from pixelate_utils import pixelate_image

LEVELS = [0, 20, 40, 60, 80]
OUT = "results/corruption_examples"
os.makedirs(OUT, exist_ok=True)

# name -> (pretty label, function(img, pct))
CORRUPTIONS = [
    ("noise",      "Gaussian noise",     noisy_image),
    ("blur",       "Gaussian blur",      blur_image),
    ("jpeg",       "JPEG compression",   jpeg_image),
    ("brightness", "Low-light (dimming)", brightness_image),
    ("pixelate",   "Pixelation (low-res)", pixelate_image),
]
DATASETS = [
    ("figstep", "FigStep (SafeBench)", load_figstep),
    ("xstest",  "XSTest (safe)",       load_xstest),
    ("mmsa",    "MMSA (safe)",          load_mmsa),
]


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


def make_strip(src, fn, out_path, label=""):
    src = src.convert("RGB")
    thumb = src.copy(); thumb.thumbnail((220, 220))
    tw, th = thumb.size
    pad, top = 8, 30
    bottom = 30 if label else pad
    n = len(LEVELS)
    strip = Image.new("RGB", (n * (tw + pad) + pad, th + top + bottom), "white")
    draw = ImageDraw.Draw(strip)
    font = _load_font(18)
    for i, pct in enumerate(LEVELS):
        tile = fn(thumb, pct)
        x = pad + i * (tw + pad)
        strip.paste(tile, (x, top))
        cap = f"{pct}%"
        bb = draw.textbbox((0, 0), cap, font=font)
        draw.text((x + (tw - (bb[2] - bb[0])) // 2, 6), cap, fill="black", font=font)
    if label:
        draw.text((pad, th + top + 6), label, fill="#444", font=font)
    strip.save(out_path)
    print("Saved:", out_path, strip.size)


def pick(samples, idx):
    for i in range(idx, len(samples)):
        if samples[i].get("image") is not None:
            return samples[i]["image"], i
    raise SystemExit("No image-bearing sample found from idx %d" % idx)


def main():
    fig_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    xs_idx  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    mm_idx  = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    idxs = {"figstep": fig_idx, "xstest": xs_idx, "mmsa": mm_idx}

    for key, pretty_ds, loader in DATASETS:
        print("Loading %s ..." % pretty_ds)
        img, i = pick(loader(), idxs[key])
        print("  using sample idx %d" % i)
        for cname, cpretty, fn in CORRUPTIONS:
            make_strip(img, fn, f"{OUT}/{cname}_strip_{key}.png",
                       "%s — %s 0%% to 80%%" % (pretty_ds, cpretty))
    print("\nDone. Pull results/corruption_examples/*.png to your Mac.")


if __name__ == "__main__":
    main()
