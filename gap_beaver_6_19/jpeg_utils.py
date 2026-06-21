"""
jpeg_utils.py — Percentage-based JPEG compression (0% = clean, 100% = max artifacts).

Realistic corruption: every image on the web is JPEG-compressed. Unlike blur/noise,
JPEG largely PRESERVES high-contrast text (so a typographic FigStep attack survives),
which makes it a useful contrast case: if safety still degrades under JPEG, it is not
simply "the model can't read the attack any more".

Percentage maps to JPEG quality (higher pct = lower quality = more blocking):

    quality = 95 - (pct/100) * 90      # pct20->77, 40->59, 60->41, 80->23

0% returns the image untouched (clean baseline, shared with the other sweeps).
"""
import io
from PIL import Image


def jpeg_quality(jpeg_pct: float) -> int:
    return max(2, round(95 - (jpeg_pct / 100.0) * 90))


def jpeg_image(img: Image.Image, jpeg_pct: float) -> Image.Image:
    """Return img re-encoded as JPEG at the quality implied by jpeg_pct
    (0 = clean/untouched, higher = blockier)."""
    if jpeg_pct <= 0:
        return img.copy()
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality(jpeg_pct))
    buf.seek(0)
    return Image.open(buf).convert("RGB")
