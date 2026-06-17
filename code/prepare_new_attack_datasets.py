#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_new_attack_datasets.py — materialize the three new image-based safety
attack datasets to LOCAL images + a keyed JSON, so the GPU eval jobs run fully
offline exactly like FigStep/XSTest/MMSA do.

  SIUO          sinwang/SIUO            167  (siuo_gen.json + images/)
  BeaverTails-V PKU-Alignment/BeaverTails-V 1180 (20 categories x evaluation split)
  SPA-VL        sqrti/SPA-VL            265  (test config, 'harm' split)

WHY NOT load_dataset?  The Newton login node has a hard RLIMIT_NPROC ~100, so
the datasets library's pyarrow/torch worker threads core-dump ("Resource
temporarily unavailable"). We instead pull the exact parquet files with
hf_hub_download / snapshot_download (max_workers=1, the sanctioned download path)
and read them with SINGLE-THREADED pyarrow — no torch, no thread pools.

Run ONCE on the login node, ONLINE (this is the only step that needs internet):
    source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
    conda activate REU
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    export USE_TORCH=0 USE_TF=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python code/prepare_new_attack_datasets.py

Output (committed-friendly layout):
    datasets/new_attacks/<ds>/images/*.png
    datasets/new_attacks/<ds>/<ds>.json   keyed by idx:
        { "<idx>": {idx, dataset, prompt, image_path, category, label:"harmful"} }
"""
import io
import os
import sys
import json
import shutil

# Keep every threadpool at 1 BEFORE importing pyarrow so it cannot exceed the
# login node's process cap.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "USE_TORCH", "USE_TF"):
    os.environ.setdefault(_v, "1" if "NUM_THREADS" in _v else "0")

# The login node's RLIMIT_NPROC + low mem make the Rust xet / hf_transfer download
# backends abort ("memory allocation of N bytes failed", Rust backtrace). FORCE the
# plain single-threaded Python downloader. Must be set before importing huggingface_hub.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_XET_DISABLE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq           # noqa: E402
import pyarrow                          # noqa: E402
from PIL import Image                   # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

pyarrow.set_cpu_count(1)
pyarrow.set_io_thread_count(1)

OUT_ROOT = os.path.join("datasets", "new_attacks")

# Expected sample counts (fail loudly on mismatch — repo convention).
EXPECT = {"siuo": 167, "beavertails": 1180, "spavl": 265}

BEAVERTAILS_CATEGORIES = [
    "animal_abuse", "dangerous_behavior", "deception_in_personal_relationships",
    "discriminatory_depictions", "environmental_damage", "false_information",
    "financial_and_academic_fraud", "hacking_or_digital_crime",
    "harmful_health_content", "horror_and_gore",
    "identity_misuse_and_impersonation", "insulting_and_harassing_behavior",
    "pornographic_content", "privacy_invasion_and_surveillance",
    "psychological_harm_and_manipulation", "psychological_horror_and_dark_themes",
    "sensitive_information_in_key_areas", "sexual_crimes",
    "terrorism_or_extremism", "violence_and_physical_harm",
]


def _fresh_dir(ds):
    d = os.path.join(OUT_ROOT, ds)
    img = os.path.join(d, "images")
    if os.path.isdir(d):
        shutil.rmtree(d)         # rebuild from scratch — never half-merge
    os.makedirs(img, exist_ok=True)
    return d, img


def _save_image(cell, path):
    """cell is an HF Image value read from parquet: {'bytes':..., 'path':...} or raw bytes."""
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None and cell.get("path") and os.path.exists(cell["path"]):
            Image.open(cell["path"]).convert("RGB").save(path)
            return
    else:
        data = cell
    if not data:
        raise ValueError("empty image bytes for %s" % path)
    Image.open(io.BytesIO(data)).convert("RGB").save(path)


def _write_json(ds, records):
    n = len(records)
    exp = EXPECT[ds]
    assert n == exp, "%s: expected %d samples, materialized %d — aborting." % (ds, exp, n)
    keyed = {}
    for r in records:
        k = str(r["idx"])
        assert k not in keyed, "duplicate idx %r in %s" % (k, ds)
        keyed[k] = r
    out = os.path.join(OUT_ROOT, ds, "%s.json" % ds)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(keyed, f, indent=2, ensure_ascii=False)
    print("  [%s] wrote %d samples -> %s" % (ds, n, out), flush=True)


# ── SIUO ────────────────────────────────────────────────────────────────────
def prepare_siuo():
    print("\n=== SIUO (sinwang/SIUO) ===", flush=True)
    d, img_dir = _fresh_dir("siuo")
    gen = hf_hub_download("sinwang/SIUO", "siuo_gen.json", repo_type="dataset")
    items = json.load(open(gen, encoding="utf-8"))
    snapshot_download("sinwang/SIUO", repo_type="dataset",
                      allow_patterns=["images/*"], max_workers=1,
                      local_dir=os.path.join(d, "_hf"))
    src_img_root = os.path.join(d, "_hf", "images")

    records = []
    for it in items:
        idx = it["question_id"]
        src = os.path.join(src_img_root, it["image"])
        dst = os.path.join(img_dir, "%s.png" % idx)
        Image.open(src).convert("RGB").save(dst)
        records.append({
            "idx": idx, "dataset": "SIUO", "prompt": it["question"],
            "image_path": os.path.abspath(dst),
            "category": it.get("category", ""), "label": "harmful",
        })
    shutil.rmtree(os.path.join(d, "_hf"), ignore_errors=True)
    _write_json("siuo", records)


# ── BeaverTails-V ───────────────────────────────────────────────────────────
def prepare_beavertails():
    print("\n=== BeaverTails-V (PKU-Alignment/BeaverTails-V) ===", flush=True)
    d, img_dir = _fresh_dir("beavertails")
    records = []
    for cat in BEAVERTAILS_CATEGORIES:
        pqf = hf_hub_download("PKU-Alignment/BeaverTails-V",
                              "data/%s/evaluation.parquet" % cat, repo_type="dataset")
        tbl = pq.read_table(pqf, use_threads=False).to_pylist()
        for i, row in enumerate(tbl):
            idx = "%s_%03d" % (cat, i)
            dst = os.path.join(img_dir, "%s.png" % idx)
            _save_image(row["image"], dst)
            records.append({
                "idx": idx, "dataset": "BeaverTails-V", "prompt": row["question"],
                "image_path": os.path.abspath(dst),
                "category": row.get("category", cat), "label": "harmful",
            })
        print("    %-40s %4d" % (cat, len(tbl)), flush=True)
    _write_json("beavertails", records)


# ── SPA-VL ──────────────────────────────────────────────────────────────────
def prepare_spavl():
    print("\n=== SPA-VL (sqrti/SPA-VL, test/harm) ===", flush=True)
    d, img_dir = _fresh_dir("spavl")
    pqf = hf_hub_download("sqrti/SPA-VL", "test/harm-00000-of-00001.parquet",
                          repo_type="dataset")
    tbl = pq.read_table(pqf, use_threads=False).to_pylist()
    records = []
    for i, row in enumerate(tbl):
        idx = "spavl_%04d" % i
        dst = os.path.join(img_dir, "%s.png" % idx)
        _save_image(row["image"], dst)
        cat = row.get("class1") or row.get("class2") or ""
        records.append({
            "idx": idx, "dataset": "SPA-VL", "prompt": row["question"],
            "image_path": os.path.abspath(dst),
            "category": cat, "label": "harmful",
        })
    _write_json("spavl", records)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    prepare_siuo()
    prepare_spavl()
    prepare_beavertails()
    print("\nAll three datasets materialized under %s/" % OUT_ROOT, flush=True)
    print("Next (GPU jobs, offline):  bash slurm_scripts/submit_new_attacks.sh", flush=True)


if __name__ == "__main__":
    main()
