"""
blur_utils.py — Percentage-based Gaussian blur (0% = clean, 100% = destroyed).

Used for BOTH the example-image strips and the eval sweeps so the blur applied
in the plots exactly matches the blur shown in the presentation images.

blur_pct -> PIL GaussianBlur radius, scaled to image size so 100% destroys any
image regardless of resolution:

    radius = (blur_pct / 100) * MAX_FRAC * min(width, height)

MAX_FRAC = 0.10 means at 100% the radius is 10% of the shorter side
(e.g. 76 px on a 760x760 image) — a featureless blur "nothingness".
"""
from PIL import Image, ImageFilter

MAX_FRAC = 0.10

def blur_radius(img: Image.Image, blur_pct: float) -> float:
    w, h = img.size
    return (blur_pct / 100.0) * MAX_FRAC * min(w, h)

def blur_image(img: Image.Image, blur_pct: float) -> Image.Image:
    """Return img blurred at the given percentage (0 = clean, 100 = nothingness)."""
    if blur_pct <= 0:
        return img.copy()
    r = blur_radius(img, blur_pct)
    return img.filter(ImageFilter.GaussianBlur(radius=r))
