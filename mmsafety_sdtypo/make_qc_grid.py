#!/usr/bin/env python
"""
make_qc_grid.py — apply all 5 corruptions to 3 sample images (idx 0 / 50 / 150)
and save a labeled 3-row x 5-col QC grid for visual inspection BEFORE any model
runs. Reads parquet directly (no `datasets` lib).

  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python3 mmsafety_sdtypo/make_qc_grid.py
"""
import os
import io
import sys
import glob
import argparse
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
import pyarrow.parquet as pq                # noqa: E402
from PIL import Image, ImageDraw            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corruption_utils import CORRUPTIONS    # noqa: E402

SUBSETS = [("Sex", 109), ("Physical_Harm", 144)]
ORDER = ["clean", "blur", "noise", "jpeg", "occlusion"]
LABELS = {"clean": "clean", "blur": "blur r=4.0", "noise": "noise std=45",
          "jpeg": "jpeg q=15", "occlusion": "occlusion 40%"}
PICKS = [0, 50, 150]   # 0=Sex, 50=Sex, 150=Physical_Harm


def find_parquet(sub):
    return glob.glob(os.path.expanduser("~/.cache/huggingface/**/%s/SD_TYPO.parquet" % sub),
                     recursive=True)[0]


def load_bytes():
    out = []
    for sub, _ in SUBSETS:
        t = pq.read_table(find_parquet(sub))
        out.extend(t.column("image").to_pylist())   # raw bytes, idx 0..252
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/ch169788/mmsafety_sdtypo")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    blobs = load_bytes()
    cell, top = 256, 28
    grid = Image.new("RGB", (cell * 5, top + cell * 3), "white")
    draw = ImageDraw.Draw(grid)
    for c, name in enumerate(ORDER):
        draw.text((c * cell + 4, 8), LABELS[name], fill="black")

    for r, idx in enumerate(PICKS):
        base = Image.open(io.BytesIO(blobs[idx])).convert("RGB")
        for c, name in enumerate(ORDER):
            corr = CORRUPTIONS[name](base).convert("RGB").resize((cell, cell))
            grid.paste(corr, (c * cell, top + r * cell))
        sub = "Sex" if idx < 109 else "Physical_Harm"
        draw.text((4, top + r * cell + 4), "idx %d (%s)" % (idx, sub), fill="red")

    out = os.path.join(args.out_dir, "qc_corruption_grid.png")
    grid.save(out)
    print("\nQC grid saved. Inspect %s before proceeding.\n" % out)


if __name__ == "__main__":
    main()