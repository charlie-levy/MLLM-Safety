"""
brightness_utils.py — Percentage-based darkening / low-light (0% = clean, 100% = near-black).

Realistic corruption: a deployed model often sees under-exposed, low-light images
(night, bad cameras). Like JPEG, dimming largely PRESERVES the shape of high-contrast
text, so it is a "text survives" corruption — a useful contrast to blur/pixelation.

Percentage maps to a brightness multiplier (higher pct = darker):

    factor = 1 - (pct/100) * 0.9       # pct20->0.82, 40->0.64, 60->0.46, 80->0.28

0% returns the image untouched (clean baseline, shared with the other sweeps).
"""
from PIL import Image, ImageEnhance

MAX_DIM = 0.9   # at 100% the image keeps only 10% of its brightness


def brightness_factor(bright_pct: float) -> float:
    return 1.0 - (bright_pct / 100.0) * MAX_DIM


def brightness_image(img: Image.Image, bright_pct: float) -> Image.Image:
    """Return img dimmed toward black at the given percentage
    (0 = clean/untouched, higher = darker)."""
    if bright_pct <= 0:
        return img.copy()
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(brightness_factor(bright_pct))
