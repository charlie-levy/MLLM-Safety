#!/usr/bin/env python3
"""
combine_figstep_tis_responses.py — Merge every FigStep TIS response CSV into ONE
file for inspection. Pulls the per-sample model outputs the FigStep evals saved
(responses_base_tis_*.csv) across the clean baseline, the severity sweep, and the
noise-% / blur-% sweeps, tags each row with the condition it came from, and writes
a single combined CSV.

Run ON NEWTON from the repo root (reads ./results/). Usage:
  python3 code/combine_figstep_tis_responses.py
Then pull just the one output file:
  scp ...:.../results/figstep_tis_all_responses.csv results_newton/
"""
import os, glob, csv

R = "results"
TAG = "base_tis"
OUT = f"{R}/figstep_tis_all_responses.csv"

# (folder, glob pattern) for each place FigStep TIS responses are written.
SOURCES = [
    (f"{R}/figstep_noise_sweep", f"responses_{TAG}_clean.csv"),
    (f"{R}/figstep_noise_sweep", f"responses_{TAG}_gaussian_noise_sev*.csv"),
    (f"{R}/figstep_noise_pct",   f"responses_{TAG}_gaussian_noise_pct_p*.csv"),
    (f"{R}/figstep_blur_pct",    f"responses_{TAG}_gaussian_blur_pct_p*.csv"),
]

def condition_from(fname):
    """Human-readable condition label derived from the file name."""
    base = os.path.basename(fname).replace(f"responses_{TAG}_", "").replace(".csv", "")
    if base == "clean":
        return "clean (0%)"
    if base.startswith("gaussian_noise_pct_p"):
        return f"noise {base.split('_p')[-1]}%"
    if base.startswith("gaussian_blur_pct_p"):
        return f"blur {base.split('_p')[-1]}%"
    if base.startswith("gaussian_noise_sev"):
        return f"noise sev{base.split('sev')[-1]}"
    if base.startswith("gaussian_blur_sev"):
        return f"blur sev{base.split('sev')[-1]}"
    return base

rows, header, n_files = [], None, 0
for folder, pat in SOURCES:
    for path in sorted(glob.glob(os.path.join(folder, pat))):
        cond = condition_from(path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = ["condition"] + reader.fieldnames
            for r in reader:
                r["condition"] = cond
                rows.append(r)
        n_files += 1
        print(f"  + {path}  ({cond})")

if not rows:
    raise SystemExit("No FigStep TIS response CSVs found. Run the FigStep evals first.")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"\nMerged {n_files} files -> {OUT}  ({len(rows)} rows)")
