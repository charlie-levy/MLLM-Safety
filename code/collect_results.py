#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_results.py — gather ALL of this study's results into ONE neat folder so a
single rsync brings everything to the laptop.

Collects three result sets, each with its FULL per-sample data (model responses,
judge inputs/outputs, judged files) + the aggregate summary:

  results/collected/
    MASTER_SUMMARY.md                 # the 3 report tables, captured verbatim
    all_numbers.csv                   # every aggregate metric, flat
    task1_msr_corruptions/            # MSR-Align under JPEG + motion blur (@20/40)
    task2_new_attacks/                # SIUO / SPA-VL / BeaverTails ASR (base vs TIS)
    msr_align_blur/                   # MSR-Align Gaussian-blur multirun (clean/20/40)

Re-runs the three report scripts first so the summaries are fresh, then copies.
CPU-only — safe on the login node (REU env). Run from the repo root:

  python code/report_msr_multirun.py     # if not already run (MSR-blur numbers)
  python code/collect_results.py
"""
import os
import sys
import csv
import json
import shutil
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join("results", "collected")

# (heading, report script, summary json it writes)
REPORTS = [
    ("MSR-Align — Gaussian-blur multirun (clean / blur20 / blur40), 3-run mean+/-std",
     "report_msr_multirun.py", "results/msr_guard_multirun_summary.json"),
    ("Task 1 — MSR-Align under JPEG + motion blur (@20/40)",
     "report_msr_corruptions.py", "results/msr_corruptions_summary.json"),
    ("Task 2 — SIUO / SPA-VL / BeaverTails ASR (base vs TIS)",
     "report_new_attacks.py", "results/new_attacks/new_attacks_summary.json"),
    ("Base — Llama-3.2-11B-Vision-Instruct (no safety, string-match) severity 0-5",
     "report_base_vision.py", "results/base_vision_eval/summary.json"),
    ("VLGuard — LLaVA-1.5-7B mixed/posthoc (Llama Guard ASR + LLaMA ORR)",
     "report_vlguard.py", "results/vlguard_eval/summary.json"),
]


def _pick_examples(items, n=3):
    """Pick up to n VARIED examples: first one per distinct category, then fill
    with distinct prompts. Avoids BeaverTails' habit of repeating one question
    across several images (which made examples 0 and 1 identical)."""
    picked, seen_cat, seen_prompt = [], set(), set()
    for it in items:                                   # pass 1: new category
        c, p = it.get("category", ""), it.get("prompt", "")
        if c not in seen_cat and p not in seen_prompt:
            picked.append(it); seen_cat.add(c); seen_prompt.add(p)
            if len(picked) >= n:
                return picked
    for it in items:                                   # pass 2: new prompt
        p = it.get("prompt", "")
        if p not in seen_prompt:
            picked.append(it); seen_prompt.add(p)
            if len(picked) >= n:
                return picked
    return picked


def copy_in(src, dst):
    """Copy a file or whole dir into dst; warn (don't crash) if src is missing."""
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print("  copied dir   %s" % src)
    elif os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print("  copied file  %s" % src)
    else:
        print("  [skip missing] %s" % src)


def run_reports():
    """Run each report so its summary.json is fresh, and print its table to stdout.
    No .md is written — the curated FINAL_REPORT.md + all_numbers.csv are the keepers."""
    for _heading, script, _ in REPORTS:
        print("\n=== running %s ===" % script, flush=True)
        try:
            p = subprocess.run([sys.executable, os.path.join("code", script)],
                               capture_output=True, text=True)
            out = p.stdout + (("\n[stderr]\n" + p.stderr) if p.returncode else "")
        except Exception as e:
            out = "ERROR running %s: %r" % (script, e)
        print(out)


def gather_full_data():
    # ── Task 1: MSR-Align JPEG + motion blur ─────────────────────────────────
    t1 = os.path.join(OUT, "task1_msr_corruptions")
    for cond in ("jpeg20", "jpeg40", "motion_blur20", "motion_blur40"):
        copy_in(os.path.join("results", "msr_guard_eval", cond),
                os.path.join(t1, cond))
    for corr in ("jpeg", "motion_blur"):
        copy_in(os.path.join("results", "sqa_%s_pct" % corr),
                os.path.join(t1, "sqa_%s" % corr))
    copy_in("results/msr_corruptions_summary.json", os.path.join(t1, "summary.json"))

    # ── Task 2: new attack datasets ──────────────────────────────────────────
    t2 = os.path.join(OUT, "task2_new_attacks")
    for ds in ("siuo", "spavl", "beavertails"):
        copy_in(os.path.join("results", "new_attacks", ds), os.path.join(t2, ds))
    copy_in("results/new_attacks/new_attacks_summary.json",
            os.path.join(t2, "summary.json"))

    # ── MSR-Align Gaussian-blur multirun ─────────────────────────────────────
    mb = os.path.join(OUT, "msr_align_blur")
    for run in (1, 2, 3):
        for cond in ("clean", "blur20", "blur40"):
            copy_in(os.path.join("results", "msr_guard_eval_run%d" % run, cond),
                    os.path.join(mb, "run%d" % run, cond))
    copy_in("results/msr_guard_multirun_summary.json", os.path.join(mb, "summary.json"))
    # MSR-blur SQA (single-run, deterministic)
    copy_in("results/sqa_noise_sweep/judged_base_msr_clean.json",
            os.path.join(mb, "sqa", "judged_base_msr_clean.json"))
    for p in (20, 40):
        copy_in("results/sqa_blur_pct/judged_base_msr_gaussian_blur_pct_p%d.json" % p,
                os.path.join(mb, "sqa", "judged_base_msr_gaussian_blur_pct_p%d.json" % p))

    # ── Base Llama-3.2-11B-Vision-Instruct (no safety; string-match; sev 0-5) ──
    copy_in("results/base_vision_eval", os.path.join(OUT, "base_vision"))

    # ── VLGuard LLaVA-1.5-7B (mixed / posthoc): Guard ASR + LLaMA ORR + SQA ────
    copy_in("results/vlguard_eval", os.path.join(OUT, "vlguard"))

    # ── A few example images + prompts from the three new attack datasets ─────
    ex = os.path.join(OUT, "dataset_examples")
    for ds in ("siuo", "spavl", "beavertails"):
        djson = os.path.join("datasets", "new_attacks", ds, "%s.json" % ds)
        if not os.path.exists(djson):
            print("  [skip missing] %s" % djson)
            continue
        with open(djson, encoding="utf-8") as f:
            items = _pick_examples(list(json.load(f).values()), 3)
        dst = os.path.join(ex, ds)
        os.makedirs(dst, exist_ok=True)
        meta = []
        for it in items:
            ip = it.get("image_path", "")
            if ip and os.path.exists(ip):
                shutil.copy2(ip, os.path.join(dst, os.path.basename(ip)))
            meta.append({"idx": it.get("idx"), "image": os.path.basename(ip),
                         "prompt": it.get("prompt", ""), "category": it.get("category", "")})
        with open(os.path.join(dst, "examples.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("  examples     %s (%d imgs)" % (ds, len(meta)))


def _rows_from_summaries():
    """Flatten the three summary JSONs into (group, item, model, metric, value) rows."""
    rows = []

    s = _load("results/msr_corruptions_summary.json")
    if s:
        for cond, m in s.items():
            for metric, val in m.items():
                rows.append(("task1_msr_corruptions", cond, "base+MSR", metric, val))

    s = _load("results/new_attacks/new_attacks_summary.json")
    if s:
        for ds, models in s.items():
            for model, m in models.items():
                for metric, val in m.items():
                    rows.append(("task2_new_attacks", ds, model, metric, val))

    s = _load("results/msr_guard_multirun_summary.json")
    if s:
        for cond, m in s.items():
            for metric, val in m.items():
                if isinstance(val, dict):           # {mean, std, raw}
                    rows.append(("msr_align_blur", cond, "base+MSR",
                                 metric + "_mean", val.get("mean")))
                    rows.append(("msr_align_blur", cond, "base+MSR",
                                 metric + "_std", val.get("std")))
                else:
                    rows.append(("msr_align_blur", cond, "base+MSR", metric, val))

    # Base-vision + VLGuard summaries vary in shape — generic dotted-path flatten.
    for group, path in (("base_vision", "results/base_vision_eval/summary.json"),
                        ("vlguard",     "results/vlguard_eval/summary.json")):
        s = _load(path)
        if s:
            _flatten_generic(group, s, rows)
    return rows


def _flatten_generic(group, obj, rows, prefix=""):
    """Append (group, dotted_path, '', '', value) rows for every leaf in obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_generic(group, v, rows, ("%s.%s" % (prefix, k)) if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten_generic(group, v, rows, "%s[%d]" % (prefix, i))
    else:
        rows.append((group, prefix, "", "", obj))


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv():
    rows = _rows_from_summaries()
    out = os.path.join(OUT, "all_numbers.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "item", "model", "metric", "value"])
        w.writerows(rows)
    print("\n  wrote %s (%d rows)" % (out, len(rows)))


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    run_reports()
    print("\n=== copying full per-sample data ===")
    gather_full_data()
    write_csv()

    size = subprocess.run(["du", "-sh", OUT], capture_output=True, text=True).stdout.strip()
    print("\n" + "=" * 70)
    print("Collected everything into %s/" % OUT)
    print("  size: %s" % (size or "?"))
    print("Pull to laptop:")
    print("  rsync -avz <user>@newton:%s/ \\" % os.path.abspath(OUT))
    print("    /Users/charlielevy/Desktop/REU/llava_cot_eval/results_6_18/")
    print("=" * 70)


if __name__ == "__main__":
    main()
