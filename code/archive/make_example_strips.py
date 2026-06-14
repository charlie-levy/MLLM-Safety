#!/usr/bin/env python3
"""
make_example_strips.py - Generate 0%..80% example strips (Gaussian NOISE and
Gaussian BLUR) for one FigStep, one XSTest (safe subset), and one MMSA (safe
subset) image, using the SAME dataset loaders the evals use (so the images are
exactly what the models see). Strips stop at 80% to match the bar-chart columns.

Run ON NEWTON from the repo root (the benchmark images live there).

Usage:
  python3 code/make_example_strips.py [fig_idx] [xstest_idx] [mmsa_idx]
  (indices default to 0; bump them to pick a different / clearer example)

Writes to results/noise_examples/. Pull those PNGs to your Mac for the slides.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))   # repo root

from dataset_loader import load_figstep, load_xstest, load_mmsa
from make_noise_strip import make_strip_from_image as noise_strip
from make_blur_strip import make_strip_from_image as blur_strip

OUT = "results/noise_examples"
os.makedirs(OUT, exist_ok=True)

def pick(samples, idx):
    """Return the first sample at/after idx that has a real (non-None) image."""
    n = len(samples)
    for i in range(idx, n):
        if samples[i].get("image") is not None:
            return samples[i]["image"], i
    raise SystemExit("No image-bearing sample found from idx %d" % idx)

fig_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
xs_idx  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
mm_idx  = int(sys.argv[3]) if len(sys.argv) > 3 else 0

DATASETS = [
    ("figstep", "FigStep (SafeBench)", load_figstep, fig_idx),
    ("xstest",  "XSTest (safe)",       load_xstest,  xs_idx),
    ("mmsa",    "MMSA (safe)",          load_mmsa,    mm_idx),
]

for key, pretty, loader, idx in DATASETS:
    print("Loading %s ..." % pretty)
    img, i = pick(loader(), idx)
    noise_strip(img, f"{OUT}/noise_strip_{key}.png",
                "%s - Gaussian noise 0%% to 80%%" % pretty)
    blur_strip(img,  f"{OUT}/blur_strip_{key}.png",
               "%s - Gaussian blur 0%% to 80%%" % pretty)
    print("  used %s sample idx %d" % (pretty, i))

print("\nDone. Pull results/noise_examples/*.png to your Mac.")
