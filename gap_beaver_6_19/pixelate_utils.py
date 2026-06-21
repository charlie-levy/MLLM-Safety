"""
pixelate_utils.py — Percentage-based pixelation / low-resolution (0% = clean, 100% = destroyed).

Realistic corruption: thumbnails, low-res webcams, and downscaled uploads. We shrink
the image then scale it back up with nearest-neighbour, producing blocky pixels.
Like blur and noise (but unlike JPEG/brightness), heavy pixelation DESTROYS small text,
so the typographic attack does not survive at high levels — a "text destroyed" corruption.

Percentage maps to the downscale fraction (higher pct = smaller intermediate = blockier):

    scale = 1 - (pct/100) * 0.95       # pct20->0.81, 40->0.62, 60->0.43, 80->0.24

0% returns the image untouched (clean baseline, shared with the other sweeps).
"""
from PIL import Image

MAX_SHRINK = 0.95   # at 100% the image is shrunk to 5% of each side before upscaling


def pixelate_image(img: Image.Image, pix_pct: float) -> Image.Image:
    """Return img pixelated at the given percentage
    (0 = clean/untouched, higher = blockier)."""
    if pix_pct <= 0:
        return img.copy()
    img = img.convert("RGB")
    w, h = img.size
    scale = 1.0 - (pix_pct / 100.0) * MAX_SHRINK
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.resize((nw, nh), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)
