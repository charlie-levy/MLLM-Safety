"""
noise_utils.py — Percentage-based Gaussian noise (0% = clean, 100% = pure static).

Used for BOTH the example-image strips and the eval sweeps so the noise applied
in the plots exactly matches the noise shown in the presentation images.

The image is alpha-blended toward a field of pure Gaussian static:

    out = (1 - a) * image + a * static,     a = noise_pct / 100

so 0% leaves the image untouched and 100% replaces it entirely with featureless
static "nothingness" (a plain additive model can't destroy high-contrast text,
so we blend toward full static instead).
"""
import numpy as np
from PIL import Image

# Gaussian static field: mid-gray mean with wide spread -> classic TV-static look
STATIC_MEAN = 128.0
STATIC_STD  = 80.0

def noisy_image(img: Image.Image, noise_pct: float, seed=None) -> Image.Image:
    """Return img blended toward Gaussian static at the given percentage
    (0 = clean, 100 = pure static). `seed` makes the result reproducible."""
    if noise_pct <= 0:
        return img.copy()
    a = min(noise_pct / 100.0, 1.0)
    rng = np.random.default_rng(seed)
    arr = np.array(img, dtype=np.float32)
    static = np.clip(rng.normal(STATIC_MEAN, STATIC_STD, arr.shape), 0, 255)
    out = (1.0 - a) * arr + a * static
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)))
