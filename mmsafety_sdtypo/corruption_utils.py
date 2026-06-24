"""
corruption_utils.py — corruption registry for the MM-SafetyBench SD_TYPO study.

NOTE on strength: the repo's legacy "severity 1-5" scale (evaluator.py) is very
mild — blur sev3 = radius 1.5, sev4 noise = std 20, and even sev5 is only
radius 2.5 / std 25 (barely visible). For this study we need corruptions strong
enough to actually degrade the image, so the registry below uses stronger explicit
parameters, tuned via the QC grid. The verbatim severity functions are kept for
provenance / comparison.

Current study levels (tune these from the QC grid):
    blur      : Gaussian radius = 4.0
    noise     : Gaussian std    = 45
    jpeg      : quality         = 15
    occlusion : block fraction  = 0.40
"""
import io
import numpy as np
from PIL import Image, ImageFilter


# ── VERBATIM from evaluator.py::_apply_corruption (mild legacy sev 1-5 scale) ───
def apply_gaussian_blur(img, severity=3):
    img_array = np.array(img, dtype=np.float32)
    radius = 0.5 + (severity - 1) * 0.5          # sev1-5 -> radius 0.5-2.5
    img_pil = Image.fromarray(np.uint8(img_array))
    return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(img, severity=4):
    img_array = np.array(img, dtype=np.float32)
    std = 5 + (severity - 1) * 5                  # sev1-5 -> std 5-25
    noise = np.random.normal(0, std, img_array.shape)
    corrupted = np.clip(img_array + noise, 0, 255)
    return Image.fromarray(np.uint8(corrupted))


# ── study-strength versions (stronger, explicit, QC-tuned) ─────────────────────
def blur_strong(img, radius=4.0):
    arr = np.array(img, dtype=np.float32)
    img_pil = Image.fromarray(np.uint8(arr))
    return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))


def noise_strong(img, std=45):
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.uint8(np.clip(arr + noise, 0, 255)))


def apply_jpeg_compression(img, quality=15):
    """JPEG compression; lower quality => heavier blocking. .copy() (PIL lazy-loads)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_occlusion(img, block_fraction=0.40, seed=42):
    """Grey block over block_fraction of the image. Fixed seed => deterministic position."""
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
    "blur":      lambda img: blur_strong(img, radius=4.0),
    "noise":     lambda img: noise_strong(img, std=45),
    "jpeg":      lambda img: apply_jpeg_compression(img, quality=15),
    "occlusion": lambda img: apply_occlusion(img, block_fraction=0.40, seed=42),
}
