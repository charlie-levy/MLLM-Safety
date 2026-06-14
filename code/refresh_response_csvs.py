#!/usr/bin/env python3
"""
refresh_response_csvs.py — Recompute the DERIVED columns of every saved response CSV
(is_refusal, is_over_refusal, conclusion, predicted_letter, correct, attack_success)
from the authoritative `full_response` text, using the FIXED detector in metrics.py.

The per-sample CSVs were written at eval time with the old (curly-apostrophe) detector,
so their derived columns are stale even though `full_response` is correct. This brings
the columns back in sync so the CSVs are safe to inspect. Metric JSONs are fixed
separately by rescore_from_responses.py.

Run on the Mac after a pull:   python3 code/refresh_response_csvs.py results_newton
Or on Newton from the repo root: python3 code/refresh_response_csvs.py
"""
import csv, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (is_refusal, is_mmsa_over_refusal, extract_answer_letter,
                     _extract_conclusion)

csv.field_size_limit(10_000_000)
B = sys.argv[1] if len(sys.argv) > 1 else "results"


def refresh(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or "full_response" not in fields:
            return 0
        rows = list(reader)
    changed = 0
    for r in rows:
        fr = r.get("full_response", "")
        new = dict(r)
        if "conclusion" in fields:
            new["conclusion"] = (_extract_conclusion(fr) or "").strip()
        if "is_refusal" in fields:
            new["is_refusal"] = is_refusal(fr)
        if "is_over_refusal" in fields:
            new["is_over_refusal"] = is_mmsa_over_refusal(fr)
        pred = extract_answer_letter(fr)
        if "predicted_letter" in fields:
            new["predicted_letter"] = pred or ""
        lbl = (r.get("label", "") or "").strip().upper()
        if "correct" in fields:
            new["correct"] = "" if not (pred and lbl) else (pred == lbl)
        if "attack_success" in fields:
            new["attack_success"] = (int(not is_refusal(fr)) if r.get("label") == "harmful" else "")
        if any(str(new[k]) != str(r[k]) for k in fields):
            changed += 1
        r.update(new)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return changed


def main():
    paths = sorted(glob.glob(f"{B}/**/responses_*.csv", recursive=True))
    combined = f"{B}/figstep_tis_all_responses.csv"
    if os.path.exists(combined):
        paths.append(combined)
    total = 0
    for p in paths:
        c = refresh(p)
        total += c
        if c:
            print(f"  refreshed {c:>4} rows  {os.path.relpath(p, B)}")
    print(f"\nRefreshed derived columns in {len(paths)} CSVs ({total} rows changed).")


if __name__ == "__main__":
    main()
