#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preflight_gap_beaver.py — CPU-only gate for the BeaverTails gap jobs.

Run on the LOGIN node before submitting. Verifies everything the GPU jobs need so
a broken job never reaches a GPU (0 GPU-hours spent on failures). Exits 0 (OK) or
1 (do NOT submit). Imports nothing heavy (no torch / pandas) — login-node safe.

Checks:
  1. self-contained pipeline files present (the 7 copied from code/), and — if
     code/ still exists — byte-identical to it (catches accidental drift).
  2. datasets/new_attacks/beavertails/beavertails.json loads and has exactly 1180
     entries, and a sample image_path exists on disk.
  3. TIS LoRA adapter dir exists (needed for the two --use_tis jobs).
  4. base model present in the HF hub cache (offline jobs can't download).
"""
import os
import sys
import json
import glob
import hashlib

if sys.version_info[0] < 3:
    sys.exit("ERROR: run with Python 3 — the login node's bare `python` is Python 2. "
             "Do `conda activate REU` first (or use `python3`).")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)          # import the SELF-CONTAINED config copy

import config                     # noqa: E402  (config.py only imports os — safe)

# scoring/corruption code: MUST be byte-identical to code/ (any drift changes the
# numbers -> hard fail). The eval driver is intentionally extended (noise + JSON),
# so it is existence-only.
MUST_MATCH = ["model_loader.py", "dataset_loader.py", "evaluator.py", "metrics.py",
              "config.py", "blur_utils.py", "noise_utils.py"]
PRESENT_ONLY = ["run_beavertails.py", "calc_asr_beavertails.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    errors, warns = [], []

    # 1) self-contained pipeline files
    for f in MUST_MATCH:
        gp = os.path.join(HERE, f)
        if not os.path.exists(gp):
            errors.append("missing pipeline file: gap_beaver_6_19/%s" % f)
            continue
        cp = os.path.join(REPO, "code", f)
        if os.path.exists(cp) and sha(gp) != sha(cp):
            errors.append("%s DIFFERS from code/%s — scoring/corruption drift, numbers "
                          "would not match the existing cells" % (f, f))
    for f in PRESENT_ONLY:
        if not os.path.exists(os.path.join(HERE, f)):
            errors.append("missing pipeline file: gap_beaver_6_19/%s" % f)

    # 2) dataset
    n_expected = config.NEW_ATTACK_COUNTS["beavertails"]   # 1180
    jpath = os.path.join(config.NEW_ATTACKS_DIR, "beavertails", "beavertails.json")
    if not os.path.exists(jpath):
        errors.append("dataset missing: %s  (run prepare_new_attack_datasets.py)" % jpath)
    else:
        data = json.load(open(jpath, encoding="utf-8"))
        if len(data) != n_expected:
            errors.append("beavertails.json has %d entries, expected %d" % (len(data), n_expected))
        else:
            print("  dataset OK: %d entries" % len(data))
        sample = next(iter(data.values()))
        ip = sample.get("image_path", "")
        if not (ip and os.path.exists(ip)):
            errors.append("sample image_path does not exist: %r" % ip)
        else:
            print("  sample image OK: %s" % ip)

    # 3) TIS adapter (the two --use_tis jobs need it)
    tis = config.TIS_LORA_PATH
    if not (os.path.isdir(tis) and os.path.exists(os.path.join(tis, "adapter_config.json"))):
        errors.append("TIS adapter missing/incomplete: %s" % tis)
    else:
        print("  TIS adapter OK: %s" % tis)

    # 4) base model cached (offline)
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    hits = glob.glob(os.path.join(hub, "models--Xkev--Llama-3.2V-11B-cot"))
    if not hits:
        warns.append("base model 'Xkev/Llama-3.2V-11B-cot' not found in %s — offline job may fail" % hub)
    else:
        print("  base model cached OK")

    for w in warns:
        print("  [warn] %s" % w)
    if errors:
        print("\nPREFLIGHT FAILED:")
        for e in errors:
            print("   - %s" % e)
        sys.exit(1)
    print("\nPREFLIGHT OK — safe to submit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
