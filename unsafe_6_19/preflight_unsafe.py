#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preflight_unsafe.py — CPU-only pre-submission validator for the unsafe-images job.

Run BEFORE submitting any GPU job. Costs 0 GPU-hours. Verifies:
  1. the dataset JSON loads and every image_path exists,
  2. the LLaVA-CoT base weights + TIS adapter + Llama Guard weights are present,
so a misconfigured run never reaches a GPU.

  python unsafe_6_19/preflight_unsafe.py            # check all 100
  python unsafe_6_19/preflight_unsafe.py --limit 1  # check just the TEST subset

Exit 0 = safe to submit. Exit 1 = a check FAILED, do NOT submit.
"""
import os
import sys
import json
import glob
import argparse
import io

# Single-thread before any heavy import (login-node proc cap); we only use stdlib here.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)

DATASET_JSON = "unsafe_6_19/unsafe_beavertails_v.json"
TIS_DIRS = ["model_weights/llama_cot_tis",
            os.path.expanduser("~/llava_cot_eval/model_weights/llama_cot_tis")]
HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
BASE_CACHE = "models--Xkev--Llama-3.2V-11B-cot"
GUARD_CACHE = "models--meta-llama--Llama-Guard-3-11B-Vision"

oks, fails = [], []


def check(name, ok, detail=""):
    (oks if ok else fails).append(name)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  — " + detail) if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("=" * 60)
    print("  PREFLIGHT — unsafe images job (CPU-only, 0 GPU-hours)")
    print("=" * 60)

    # 1. dataset json + images
    if not os.path.exists(DATASET_JSON):
        check("dataset JSON exists", False, DATASET_JSON); _summary()
    with io.open(DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = sorted(data, key=lambda x: int(x))
    if args.limit:
        keys = keys[:args.limit]
    check("dataset JSON loads (%d entries)" % len(keys), len(keys) > 0)
    missing = [data[k]["image_path"] for k in keys if not os.path.exists(data[k]["image_path"])]
    check("all %d image_paths exist" % len(keys), not missing,
          "" if not missing else "missing %d, e.g. %s" % (len(missing), missing[:2]))
    same_prompt = len({data[k]["prompt"] for k in keys}) == 1
    check("prompt present + consistent", same_prompt)

    # 2. model weights
    check("TIS adapter dir", any(os.path.isdir(d) for d in TIS_DIRS),
          "looked in: %s" % TIS_DIRS)
    check("base LLaVA-CoT in HF cache", os.path.isdir(os.path.join(HF_HUB, BASE_CACHE)),
          os.path.join(HF_HUB, BASE_CACHE))
    check("Llama Guard 3 Vision in HF cache", os.path.isdir(os.path.join(HF_HUB, GUARD_CACHE)),
          os.path.join(HF_HUB, GUARD_CACHE))

    _summary()


def _summary():
    print("-" * 60)
    if fails:
        print("  RESULT: FAILED (%d) -> DO NOT SUBMIT: %s" % (len(fails), ", ".join(fails)))
        sys.exit(1)
    print("  RESULT: ALL PASSED (%d) -> safe to submit." % len(oks))
    sys.exit(0)


if __name__ == "__main__":
    main()
