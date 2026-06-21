"""
motion_blur_utils.py — Percentage-based motion blur (0% = clean, 100% = heavy streak).

Realistic corruption: a moving camera or moving subject smears the image along a
direction. Unlike Gaussian blur (which is isotropic / symmetric), motion blur is
DIRECTIONAL — it averages the image along a line. Like Gaussian blur and pixelate
(but unlike JPEG/brightness) it DESTROYS small text once the streak is long enough,
so a typographic attack stops surviving at high levels.

Percentage maps to the streak length as a fraction of the smaller image side:

    length = round( (pct/100) * 0.15 * min(H, W) )    # pct20->~3% , 40->~6% of the side

0% (or a sub-2px streak) returns the image untouched (clean baseline, shared with
the other sweeps). Deterministic: a fixed 45° streak, no randomness, so greedy
eval stays reproducible.
"""
import math

import numpy as np
from PIL import Image

MAX_FRAC = 0.15      # at 100% the streak spans 15% of the shorter side
ANGLE_DEG = 45.0     # fixed diagonal streak — deterministic, no RNG


def motion_blur_length(pct: float, min_side: int) -> int:
    """Streak length in pixels for a given percentage and image size."""
    return int(round((pct / 100.0) * MAX_FRAC * min_side))


def motion_blur_image(img: Image.Image, pct: float) -> Image.Image:
    """Return img motion-blurred at the given percentage
    (0 = clean/untouched, higher = longer directional streak)."""
    if pct <= 0:
        return img.copy()

    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]

    length = motion_blur_length(pct, min(h, w))
    if length < 2:                      # too short to see — treat as clean
        return img.copy()

    dx = math.cos(math.radians(ANGLE_DEG))
    dy = math.sin(math.radians(ANGLE_DEG))

    # Average `length` copies of the image shifted along the streak direction,
    # centred so the blur is symmetric about each pixel.
    acc = np.zeros_like(arr)
    for t in range(length):
        off = t - (length - 1) / 2.0
        sx = int(round(off * dx))
        sy = int(round(off * dy))
        acc += np.roll(np.roll(arr, sy, axis=0), sx, axis=1)
    acc /= length

    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))
