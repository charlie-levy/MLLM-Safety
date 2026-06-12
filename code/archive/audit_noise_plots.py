#!/usr/bin/env python3
"""
audit_noise_plots.py — Print the exact numbers feeding the per-model NOISE plots
(plot_per_model_noise) for TIS, SAGE, MSR, so accuracy can be verified at a glance.

For each model it prints ASR / ORR / SQA at severity 0 (clean) .. 5 under gaussian
noise, reading the SAME files the plot reads. Missing cells show as 'pending'.

Usage: python3 code/audit_noise_plots.py
"""
import json, os

BASE = "results_newton"

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

def get_asr(d):
    if d is None: return None
    return d.get("asr_pct", d.get("asr"))

def fmt(v):
    return f"{v:5.1f}" if v is not None else "  ·  "

MODELS = [("base_tis", "TIS"), ("base_sage", "SAGE"), ("base_msr", "MSR-Align")]

for tag, name in MODELS:
    print("=" * 56)
    print(f"  {name}  — metrics vs gaussian-noise severity")
    print("=" * 56)
    print(f"  {'sev':>3} | {'ASR':>5} | {'ORR':>5} | {'SQA':>5}   (file source)")
    print("  " + "-" * 52)

    complete = True
    for s in range(0, 6):
        if s == 0:
            asr = get_asr(load(f"{BASE}/figstep_noise_sweep/asr_{tag}_clean.json"))
            orr_d = load(f"{BASE}/orr/orr_{tag}.json")
            orr = orr_d["avg_orr_pct"] if orr_d else None
            sqa_d = load(f"{BASE}/sqa_noise_sweep/judged_{tag}_clean.json")
            sqa = sqa_d["accuracy"] if sqa_d else None
            src = "clean"
        else:
            asr = get_asr(load(f"{BASE}/figstep_noise_sweep/asr_{tag}_gaussian_noise_sev{s}.json"))
            orr_d = load(f"{BASE}/orr_noise_sweep/orr_{tag}_gaussian_noise_sev{s}.json")
            orr = orr_d["avg_orr_pct"] if orr_d else None
            sqa_d = load(f"{BASE}/sqa_noise_sweep/judged_{tag}_gaussian_noise_sev{s}.json")
            sqa = sqa_d["accuracy"] if sqa_d else None
            src = f"noise sev{s}"

        if None in (asr, orr, sqa):
            complete = False
        print(f"  {s:>3} | {fmt(asr)} | {fmt(orr)} | {fmt(sqa)}   ({src})")

    status = "COMPLETE — all 18 cells present" if complete else "INCOMPLETE — '·' cells still pending"
    print(f"  >> {name}: {status}\n")
