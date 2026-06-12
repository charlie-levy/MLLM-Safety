#!/usr/bin/env python3
"""
verify_blur_pct_results.py — Validate the percentage-BLUR sweep for TIS/SAGE/MSR.

Checks each cell (model x metric x {0,20,40,60,80}%) for correct model tag,
full sample counts, valid metric ranges, and low judge-unparsed. 0% reuses the
clean baselines. We stop at 80% (no 100% column).

Run ON NEWTON from repo root (reads ./results/). Usage:
  python3 code/verify_blur_pct_results.py
"""
import os, json

R = "results"
EXPECT_XSTEST, EXPECT_MMSA, EXPECT_SQA = 250, 428, 250
LEVELS = [0, 20, 40, 60, 80]
MODELS = [("base_tis", "TIS"), ("base_sage", "SAGE"), ("base_msr", "MSR")]

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

problems, asr_totals = [], {}

def asr_paths(tag, p):
    return (f"{R}/figstep_noise_sweep/asr_{tag}_clean.json" if p == 0
            else f"{R}/figstep_blur_pct/asr_{tag}_gaussian_blur_pct_p{p}.json")

def orr_paths(tag, p):
    return (f"{R}/orr_{tag}/orr_results.json" if p == 0
            else f"{R}/orr_blur_pct/orr_{tag}_gaussian_blur_pct_p{p}.json")

def sqa_paths(tag, p):
    return (f"{R}/sqa_noise_sweep/judged_{tag}_clean.json" if p == 0
            else f"{R}/sqa_blur_pct/judged_{tag}_gaussian_blur_pct_p{p}.json")

for tag, name in MODELS:
    print(f"\n=== {name} ({tag}) ===")
    for p in LEVELS:
        lbl = "clean" if p == 0 else f"{p}%"
        # ASR
        d = load(asr_paths(tag, p))
        if d is None:
            print(f"  ASR  {lbl:5}  MISSING"); problems.append(f"{name} ASR {lbl}: missing")
        else:
            mt, nt, av = d.get("model"), d.get("n_total"), d.get("asr_pct", d.get("asr"))
            asr_totals[f"{name} {lbl}"] = nt
            ok = mt == tag and nt and av is not None and 0 <= av <= 100
            print(f"  ASR  {lbl:5}  {'OK ' if ok else 'BAD'}  asr={av}  n={nt}  model={mt}")
            if mt != tag: problems.append(f"{name} ASR {lbl}: model={mt} expected {tag}")
            if not nt:    problems.append(f"{name} ASR {lbl}: bad n_total")
        # ORR
        d = load(orr_paths(tag, p))
        if d is None:
            print(f"  ORR  {lbl:5}  MISSING"); problems.append(f"{name} ORR {lbl}: missing")
        else:
            mt = d.get("model"); xn = d.get("xstest", {}).get("n_total")
            mn = d.get("mmsa_combined", {}).get("n_total"); av = d.get("avg_orr_pct")
            tag_ok = True if p == 0 else (mt == tag)
            ok = tag_ok and xn == EXPECT_XSTEST and mn == EXPECT_MMSA and av is not None
            print(f"  ORR  {lbl:5}  {'OK ' if ok else 'BAD'}  avg={av}  xstest={xn}  mmsa={mn}  model={mt}")
            if not tag_ok: problems.append(f"{name} ORR {lbl}: model={mt} expected {tag}")
            if xn != EXPECT_XSTEST: problems.append(f"{name} ORR {lbl}: XSTest n={xn}")
            if mn != EXPECT_MMSA:   problems.append(f"{name} ORR {lbl}: MMSA n={mn}")
        # SQA
        d = load(sqa_paths(tag, p))
        if d is None:
            print(f"  SQA  {lbl:5}  MISSING (judge not run?)"); problems.append(f"{name} SQA {lbl}: missing")
        else:
            tot, acc, unp = d.get("total"), d.get("accuracy"), d.get("unparsed", 0)
            ok = tot == EXPECT_SQA and acc is not None and unp <= 10
            print(f"  SQA  {lbl:5}  {'OK ' if ok else 'BAD'}  acc={acc}  n={tot}  unparsed={unp}")
            if tot != EXPECT_SQA: problems.append(f"{name} SQA {lbl}: total={tot}")
            if unp > 10:          problems.append(f"{name} SQA {lbl}: {unp} unparsed")

vals = set(v for v in asr_totals.values() if v is not None)
print("\n--- FigStep ASR n_total consistency ---")
print(f"  {'consistent: ' + str(vals.pop()) if len(vals) <= 1 else 'INCONSISTENT: ' + str(asr_totals)}")

print("\n" + "=" * 60)
if not problems:
    print("ALL CHECKS PASSED — every blur-% cell is present and valid.")
else:
    print(f"{len(problems)} PROBLEM(S):")
    for p in problems:
        print("  -", p)
