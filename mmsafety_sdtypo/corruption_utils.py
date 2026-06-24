"""
corruption_utils.py — corruption registry for the MM-SafetyBench SD_TYPO study.

gaussian_blur / gaussian_noise are COPIED VERBATIM from the existing repo
implementation used in the FigStep / BeaverTails-V severity sweeps:
    llava_cot_eval/code/evaluator.py :: Evaluator._apply_corruption
so "blur severity 3" and "noise severity 4" are exactly the values used before:
    blur  sev=3 -> radius = 0.5 + (3-1)*0.5 = 1.5
    noise sev=4 -> std    = 5   + (4-1)*5   = 20
jpeg + occlusion are written fresh (no prior repo implementation).
"""
import io
import numpy as np
from PIL import Image, ImageFilter


# ── VERBATIM from evaluator.py::_apply_corruption (severity 1-5 scale) ──────────
def apply_gaussian_blur(img, severity=3):
    img_array = np.array(img, dtype=np.float32)
    # severity 1-5 maps to radius 0.5-2.5
    radius = 0.5 + (severity - 1) * 0.5
    img_pil = Image.fromarray(np.uint8(img_array))
    return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(img, severity=4):
    img_array = np.array(img, dtype=np.float32)
    # severity 1-5 maps to std 5-25
    std = 5 + (severity - 1) * 5
    noise = np.random.normal(0, std, img_array.shape)
    corrupted = np.clip(img_array + noise, 0, 255)
    return Image.fromarray(np.uint8(corrupted))


# ── fresh (spec-provided) ──────────────────────────────────────────────────────
def apply_jpeg_compression(img, quality=25):
    """JPEG compression, quality=25 => heavy blocking. .copy() because PIL lazy-loads."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_occlusion(img, block_fraction=0.3, seed=42):
    """Grey block over ~30% of the image. Fixed seed => deterministic block position."""
    rng = np.random.RandomState(seed)
    arr = np.array(img.convert("RGB")).copy()
    h, w = arr.shape[:2]
    bh = int(h * block_fraction)
    bw = int(w * block_fraction)
    y = rng.randint(0, max(1, h - bh))
    x = rng.randint(0, max(1, w - bw))
    arr[y:y + bh, x:x + bw] = 128
    return Image.fromarray(arr)


CORRUPTIONS = {
    "clean":     lambda img: img.copy(),
    "blur":      lambda img: apply_gaussian_blur(img, severity=3),    # verbatim, radius 1.5
    "noise":     lambda img: apply_gaussian_noise(img, severity=4),   # verbatim, std 20
    "jpeg":      lambda img: apply_jpeg_compression(img, quality=25),
    "occlusion": lambda img: apply_occlusion(img, block_fraction=0.3, seed=42),
}