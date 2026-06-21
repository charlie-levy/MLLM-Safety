#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preflight_gap_corruptions.py — CPU-only gate for the BeaverTails-V NEW-corruption jobs
(motion_blur / jpeg / pixelate, at 20% and 40%, base + TIS = 12 cells).

Run on the LOGIN node before submitting. Verifies everything the GPU jobs need so a
broken job never reaches a GPU (0 GPU-hours on failures). Exits 0 (OK) / 1 (do NOT
submit). Imports nothing heavy (no torch / numpy) — login-node safe.

Checks (mirrors preflight_gap_beaver.py, plus the 3 corruption utils):
  1. self-contained pipeline files present, and — if code/ exists — byte-identical
     to it. The corruption math (jpeg/motion_blur/pixelate) MUST match code/ exactly
     or the perturbation (and thus the numbers) would differ from the project standard.
  2. evaluator copy actually branches on jpeg_pct / motion_blur_pct / pixelate_pct.
  3. beavertails.json loads, has 1180 entries, sample image exists.
  4. TIS LoRA adapter present (the six --use_tis jobs need it).
  5. base model present in the HF hub cache (offline jobs can't download).
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
# perturbation or the scoring -> numbers would not be comparable). The eval driver is
# intentionally extended (the COND map), so it is existence-only.
MUST_MATCH = ["model_loader.py", "dataset_loader.py", "evaluator.py", "metrics.py",
              "config.py", "blur_utils.py", "noise_utils.py",
              "jpeg_utils.py", "motion_blur_utils.py", "pixelate_utils.py"]
PRESENT_ONLY = ["run_beavertails.py", "calc_asr_corruptions.py"]

# the new corruption branches the evaluator copy must contain
NEEDED_BRANCHES = ["jpeg_pct", "motion_blur_pct", "pixelate_pct"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    errors, warns = [], []

    # 1) self-contained pipeline files (byte-identical to code/)
    for f in MUST_MATCH:
        gp = os.path.join(HERE, f)
        if not os.path.exists(gp):
            errors.append("missing pipeline file: gap_beaver_6_19/%s  (cp code/%s here)" % (f, f))
            continue
        cp = os.path.join(REPO, "code", f)
        if os.path.exists(cp) and sha(gp) != sha(cp):
            errors.append("%s DIFFERS from code/%s — corruption/scoring drift" % (f, f))
        else:
            print("  OK (identical): %s" % f)
    for f in PRESENT_ONLY:
        if not os.path.exists(os.path.join(HERE, f)):
            errors.append("missing pipeline file: gap_beaver_6_19/%s" % f)

    # 2) evaluator branches present (catches an old evaluator copy without the new types)
    ev_src = open(os.path.join(HERE, "evaluator.py"), encoding="utf-8").read()
    for b in NEEDED_BRANCHES:
        if b not in ev_src:
            errors.append("evaluator.py copy has no '%s' branch — corruption would raise" % b)

    # 3) dataset
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

    # 4) TIS adapter (the six --use_tis jobs need it)
    tis = config.TIS_LORA_PATH
    if not (os.path.isdir(tis) and os.path.exists(os.path.join(tis, "adapter_config.json"))):
        errors.append("TIS adapter missing/incomplete: %s" % tis)
    else:
        print("  TIS adapter OK: %s" % tis)

    # 5) base model cached (offline)
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    if glob.glob(os.path.join(hub, "models--Xkev--Llama-3.2V-11B-cot")):
        print("  base model cached OK")
    else:
        warns.append("base model 'Xkev/Llama-3.2V-11B-cot' not in %s — offline job may fail" % hub)

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
