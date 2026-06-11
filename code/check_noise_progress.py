#!/usr/bin/env python3
"""
check_noise_progress.py — Report which NOISE-sweep result files still need to be
produced for the per-model noise plots (TIS / SAGE / MSR).

Run ON NEWTON from the repo root (reads ./results/). Reports present vs missing
for every (model, metric, severity) cell. SQA is checked at the raw_*.jsonl
level (eval done); the judge turns those into judged_*.json afterwards.

Usage: python3 code/check_noise_progress.py
"""
import os

R = "results"

# metric -> (folder, filename_template). {tag} and {s} get filled in.
SPECS = {
    "ASR": ("figstep_noise_sweep", "asr_{tag}_gaussian_noise_sev{s}.json"),
    "ORR": ("orr_noise_sweep",     "orr_{tag}_gaussian_noise_sev{s}.json"),
    "SQA": ("sqa_noise_sweep",     "raw_{tag}_gaussian_noise_sev{s}.jsonl"),
}
# clean-baseline files (severity 0)
CLEAN = {
    "ASR": ("figstep_noise_sweep", "asr_{tag}_clean.json"),
    "ORR": ("orr",                 "orr_{tag}.json"),
    "SQA": ("sqa_noise_sweep",     "raw_{tag}_clean.jsonl"),
}

MODELS = [("base_tis", "TIS"), ("base_sage", "SAGE"), ("base_msr", "MSR")]

total_missing = 0
for tag, name in MODELS:
    missing = []
    # clean
    for metric, (folder, tmpl) in CLEAN.items():
        if not os.path.exists(os.path.join(R, folder, tmpl.format(tag=tag))):
            missing.append(f"{metric} clean")
    # severities 1-5
    for s in range(1, 6):
        for metric, (folder, tmpl) in SPECS.items():
            if not os.path.exists(os.path.join(R, folder, tmpl.format(tag=tag, s=s))):
                missing.append(f"{metric} sev{s}")

    total_missing += len(missing)
    if missing:
        print(f"{name:5} INCOMPLETE — {len(missing)} missing: {', '.join(missing)}")
    else:
        print(f"{name:5} COMPLETE ✓ (all 18 noise cells present)")

print("-" * 60)
if total_missing == 0:
    print("ALL THREE COMPLETE — judge SQA, then pull + regenerate plots.")
else:
    print(f"{total_missing} cells still pending across the three models.")
