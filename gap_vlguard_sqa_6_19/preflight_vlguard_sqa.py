#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preflight_vlguard_sqa.py — CPU gate for the VLGuard SQA-under-blur jobs.

Fills the 4 missing SQA cells (mixed/posthoc x blur20/blur40). Verifies everything
the jobs need before any GPU time. Exits 0 (OK) / 1 (do NOT submit). No heavy imports.

  conda activate REU && python gap_vlguard_sqa_6_19/preflight_vlguard_sqa.py
"""
import os
import sys
import json
import glob

if sys.version_info[0] < 3:
    sys.exit("ERROR: run with Python 3 — login `python` is Python 2. `conda activate REU` first.")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "code"))   # reuse the proven config/eval/judge

import config   # noqa: E402  (config.py imports only os — login-safe)

JUDGE = "NousResearch/Meta-Llama-3-8B-Instruct"


def main():
    errors, warns = [], []

    # 1) converted VLGuard LLaVA-1.5 weights (both variants)
    for v in sorted(config.VLGUARD_VARIANTS):
        hf = config.VLGUARD_VARIANTS[v]["hf"]
        if os.path.isdir(hf) and os.path.exists(os.path.join(hf, "config.json")):
            print("  vlguard %s weights OK: %s" % (v, hf))
        else:
            errors.append("VLGuard %s weights missing: %s  (run code/convert_vlguard_to_hf.py)" % (v, hf))

    # 2) ScienceQA-250 dataset
    sqa = "datasets/scienceqa_250.json"
    if not os.path.exists(sqa):
        errors.append("missing %s" % sqa)
    else:
        d = json.load(open(sqa, encoding="utf-8"))
        if len(d) != 250:
            errors.append("scienceqa_250.json has %d entries (expected 250)" % len(d))
        else:
            print("  ScienceQA-250 OK: 250 entries")

    # 3) judge model cached (offline)
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    if glob.glob(os.path.join(hub, "models--NousResearch--Meta-Llama-3-8B-Instruct")):
        print("  judge (%s) cached OK" % JUDGE)
    else:
        warns.append("judge model %s not in %s — offline judge may fail" % (JUDGE, hub))

    # 4) base LLaVA-1.5 processor/config cached (load_vlguard may read the HF template)
    if glob.glob(os.path.join(hub, "models--llava-hf--llava-1.5-7b-hf")):
        print("  llava-1.5 base cached OK")
    else:
        warns.append("llava-hf/llava-1.5-7b-hf not in cache — may be needed for the processor")

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
