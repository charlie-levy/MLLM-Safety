#!/usr/bin/env python3
"""
make_example_strips.py — Generate 0%..100% Gaussian-noise example strips for one
FigStep, one XSTest (safe subset), and one MMSA (safe subset) image, using the
SAME dataset loaders the evals use (so the images are exactly what the models see).

Run ON NEWTON from the repo root (the benchmark images live there).

Usage:
  python3 code/make_example_strips.py [fig_idx] [xstest_idx] [mmsa_idx]
  (indices default to 0; bump them to pick a different / clearer example)

Writes to results/noise_examples/. Pull those PNGs to your Mac for the slides.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from dataset_loader import load_figstep, load_xstest, load_mmsa
from make_noise_strip import make_strip_from_image

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

print("Loading FigStep ..."); fig = load_figstep()
img, i = pick(fig, fig_idx)
make_strip_from_image(img, f"{OUT}/noise_strip_figstep.png",
                      "FigStep (SafeBench) — Gaussian noise 0% to 100%")
print("  used FigStep sample idx", i)

print("Loading XSTest (safe subset) ..."); xs = load_xstest()
img, i = pick(xs, xs_idx)
make_strip_from_image(img, f"{OUT}/noise_strip_xstest.png",
                      "XSTest (safe) — Gaussian noise 0% to 100%")
print("  used XSTest sample idx", i)

print("Loading MMSA (safe subset) ..."); mm = load_mmsa()
img, i = pick(mm, mm_idx)
make_strip_from_image(img, f"{OUT}/noise_strip_mmsa.png",
                      "MMSA (safe) — Gaussian noise 0% to 100%")
print("  used MMSA sample idx", i)

print("\nDone. Pull results/noise_examples/*.png to your Mac.")
