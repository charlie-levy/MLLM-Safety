#!/usr/bin/env python3
"""
generate_experiments_log.py
Auto-generates results_newton/experiments_log.csv from actual result files.
Run any time new results arrive.
Usage: python3 code/generate_experiments_log.py
"""
import json, os, csv, glob

BASE = "results_newton"
OUT  = os.path.join(BASE, "experiments_log.csv")

FIELDS = [
    "exp_id", "experiment_name", "model", "eval_benchmark",
    "corruption_type", "severity",
    "asr_pct", "orr_xstest_pct", "orr_mmsa_pct", "orr_avg_pct",
    "sqa_regex_acc", "sqa_judged_acc",
    "what_it_measures", "result_file", "date_run", "status", "notes",
]

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def asr(d):
    if d is None: return ""
    return str(round(d.get("asr_pct", d.get("asr", "")), 1)) if d.get("asr_pct") is not None or d.get("asr") is not None else ""

def orr_row(d):
    if d is None: return ("", "", "")
    x = round(d["xstest"]["orr_pct"], 1) if "xstest" in d else ""
    m = round(d["mmsa_combined"]["orr_pct"], 1) if "mmsa_combined" in d else ""
    a = round(d["avg_orr_pct"], 1) if "avg_orr_pct" in d else ""
    return (str(x), str(m), str(a))

def sqa_regex(tag, sweep_dir):
    d = load(os.path.join(BASE, sweep_dir, f"acc_{tag}.json"))
    if d is None: return ""
    return str(round(d["accuracy"], 1))

def sqa_judged(tag, sweep_dir):
    d = load(os.path.join(BASE, sweep_dir, f"judged_{tag}.json"))
    if d is None: return "pending"
    return str(round(d["accuracy"], 1))

rows = []

# ── Paper reference row ──────────────────────────────────────────────────────
rows.append({
    "exp_id": "PAPER-MSR",
    "experiment_name": "MSR-Align - Published Paper Numbers (reference)",
    "model": "MSR-Align (arXiv:2506.19257)",
    "eval_benchmark": "XSTest + MMSA + FigStep",
    "corruption_type": "clean", "severity": "0",
    "asr_pct": "23.8", "orr_xstest_pct": "57.6", "orr_mmsa_pct": "76.2", "orr_avg_pct": "66.9",
    "sqa_regex_acc": "", "sqa_judged_acc": "88.7",
    "what_it_measures": "Published numbers from MSR-Align paper - use as reference",
    "result_file": "arXiv:2506.19257",
    "date_run": "2025", "status": "reference",
    "notes": "SQA utility from paper Table 3. If our CLEAN-MSR differs significantly, check adapter loading.",
})

# ── Clean baselines ───────────────────────────────────────────────────────────
for model_tag, model_name, exp_prefix in [
    ("base",      "Base LLaVA-CoT",           "CLEAN-BASE"),
    ("base_tis",  "TIS (Think-in-Safety)",     "CLEAN-TIS"),
    ("base_sage", "SAGE",                       "CLEAN-SAGE"),
    ("base_msr",  "MSR-Align",                  "CLEAN-MSR"),
]:
    orr_d  = load(os.path.join(BASE, "orr", f"orr_{model_tag}.json"))
    asr_d  = load(os.path.join(BASE, "figstep_noise_sweep", f"asr_{model_tag}_clean.json"))
    x, m, a = orr_row(orr_d)
    sqa_tag = f"{model_tag}_clean"
    regex  = sqa_regex(sqa_tag, "sqa_noise_sweep")
    judged = sqa_judged(sqa_tag, "sqa_noise_sweep")
    status = "complete" if (orr_d and asr_d) else "pending"
    rows.append({
        "exp_id": exp_prefix,
        "experiment_name": f"{model_name} - Clean Baseline",
        "model": model_name,
        "eval_benchmark": "XSTest(250) + MMSA(428) + FigStep(500) + ScienceQA(250)",
        "corruption_type": "clean", "severity": "0",
        "asr_pct": asr(asr_d), "orr_xstest_pct": x, "orr_mmsa_pct": m, "orr_avg_pct": a,
        "sqa_regex_acc": regex, "sqa_judged_acc": judged,
        "what_it_measures": "Clean image safety (ASR), over-refusal (ORR), and utility (SQA)",
        "result_file": f"orr/{model_tag}.json + figstep_noise_sweep/asr_{model_tag}_clean.json",
        "date_run": "2026-06", "status": status, "notes": "",
    })

# ── Noise sweep ───────────────────────────────────────────────────────────────
for model_tag, model_name in [("base", "Base LLaVA-CoT"), ("base_tis", "TIS (Think-in-Safety)"), ("base_sage", "SAGE")]:
    short = model_tag.upper().replace("BASE_", "").replace("BASE", "BASE")
    for sev in [1, 2, 3, 4, 5]:
        asr_d = load(os.path.join(BASE, "figstep_noise_sweep", f"asr_{model_tag}_gaussian_noise_sev{sev}.json"))
        orr_d = load(os.path.join(BASE, "orr_noise_sweep", f"orr_{model_tag}_gaussian_noise_sev{sev}.json"))
        x, m, a = orr_row(orr_d)
        tag = f"{model_tag}_gaussian_noise_sev{sev}"
        regex  = sqa_regex(tag, "sqa_noise_sweep")
        judged = sqa_judged(tag, "sqa_noise_sweep")
        status = "complete" if (asr_d and orr_d) else ("partial" if (asr_d or orr_d) else "pending")
        exp_id = f"NOISE-SEV{sev}-{short}"
        rows.append({
            "exp_id": exp_id,
            "experiment_name": f"{model_name} - Gaussian Noise Sev{sev}",
            "model": model_name,
            "eval_benchmark": "FigStep(500) + XSTest(250) + MMSA(428) + ScienceQA(250)",
            "corruption_type": "gaussian_noise", "severity": str(sev),
            "asr_pct": asr(asr_d), "orr_xstest_pct": x, "orr_mmsa_pct": m, "orr_avg_pct": a,
            "sqa_regex_acc": regex, "sqa_judged_acc": judged,
            "what_it_measures": f"Gaussian noise sev{sev} effect on safety and utility",
            "result_file": f"figstep_noise_sweep/ + orr_noise_sweep/ + sqa_noise_sweep/",
            "date_run": "2026-06", "status": status, "notes": "",
        })

# ── Blur sweep ────────────────────────────────────────────────────────────────
for model_tag, model_name in [("base", "Base LLaVA-CoT"), ("base_tis", "TIS (Think-in-Safety)")]:
    short = model_tag.upper().replace("BASE_", "").replace("BASE", "BASE")
    for sev in [1, 2, 3, 4, 5]:
        asr_d = load(os.path.join(BASE, "figstep_blur_sweep", f"asr_{model_tag}_gaussian_blur_sev{sev}.json"))
        orr_d = load(os.path.join(BASE, "orr_blur_sweep", f"orr_{model_tag}_gaussian_blur_sev{sev}.json"))
        x, m, a = orr_row(orr_d)
        tag = f"{model_tag}_gaussian_blur_sev{sev}"
        regex  = sqa_regex(tag, "sqa_blur_sweep")
        judged = sqa_judged(tag, "sqa_blur_sweep")
        status = "complete" if (orr_d) else "pending"
        exp_id = f"BLUR-SEV{sev}-{short}"
        rows.append({
            "exp_id": exp_id,
            "experiment_name": f"{model_name} - Gaussian Blur Sev{sev}",
            "model": model_name,
            "eval_benchmark": "FigStep(500) + XSTest(250) + MMSA(428) + ScienceQA(250)",
            "corruption_type": "gaussian_blur", "severity": str(sev),
            "asr_pct": asr(asr_d), "orr_xstest_pct": x, "orr_mmsa_pct": m, "orr_avg_pct": a,
            "sqa_regex_acc": regex, "sqa_judged_acc": judged,
            "what_it_measures": f"Gaussian blur sev{sev} effect on safety and utility",
            "result_file": f"figstep_noise_sweep/ + orr_blur_sweep/ + sqa_blur_sweep/",
            "date_run": "2026-06", "status": status, "notes": "",
        })

# ── Bypass examples ───────────────────────────────────────────────────────────
rows.append({
    "exp_id": "BYPASS-TIS-BLUR4",
    "experiment_name": "TIS Bypass Examples - Gaussian Blur Sev4",
    "model": "TIS (Think-in-Safety)",
    "eval_benchmark": "FigStep SafeBench (ForbidQI cat1)",
    "corruption_type": "gaussian_blur", "severity": "4",
    "asr_pct": "", "orr_xstest_pct": "", "orr_mmsa_pct": "", "orr_avg_pct": "",
    "sqa_regex_acc": "", "sqa_judged_acc": "",
    "what_it_measures": "3 specific images where TIS refuses clean but complies after blur sev4",
    "result_file": "bypass_examples/bypass_results.json",
    "date_run": "2026-06-10", "status": "complete",
    "notes": "All 3 from ForbidQI cat1: money laundering, criminal syndicates, forced labor",
})

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

print(f"Written {len(rows)} rows to {OUT}")
