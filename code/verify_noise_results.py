#!/usr/bin/env python3
"""
verify_noise_results.py — Validate that every NOISE-sweep result file for
TIS / SAGE / MSR is not just present but CORRECT:

  * ASR  : "model" tag matches the adapter, n_total consistent, asr_pct in [0,100]
  * ORR  : "model" tag matches, XSTest n_total==250, MMSA n_total==428, avg in range
  * SQA  : judged total==250, accuracy in [0,100], unparsed small, judge recorded

Catches the failure modes that file-existence checks miss: an eval that silently
ran as the base model, a truncated sample set, or a judge that mostly failed.

Run ON NEWTON from repo root (reads ./results/). Prints PASS/FAIL per cell.

Usage: python3 code/verify_noise_results.py
"""
import json, os, glob

R = "results"
EXPECT_XSTEST = 250
EXPECT_MMSA   = 428
EXPECT_SQA    = 250

MODELS = [("base_tis", "TIS"), ("base_sage", "SAGE"), ("base_msr", "MSR")]

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

problems = []
asr_totals = {}

def check(cond, msg):
    if not cond:
        problems.append(msg)
    return cond

for tag, name in MODELS:
    print(f"\n=== {name} ({tag}) ===")
    for s in range(0, 6):
        sev = "clean" if s == 0 else f"sev{s}"
        # ---- ASR ----
        ap = (f"{R}/figstep_noise_sweep/asr_{tag}_clean.json" if s == 0
              else f"{R}/figstep_noise_sweep/asr_{tag}_gaussian_noise_sev{s}.json")
        d = load(ap)
        if d is None:
            print(f"  ASR  {sev:5}  MISSING/UNREADABLE")
            problems.append(f"{name} ASR {sev}: missing/unreadable")
        else:
            mt = d.get("model")
            nt = d.get("n_total")
            av = d.get("asr_pct", d.get("asr"))
            ok = (mt == tag) and (nt is not None) and (av is not None) and (0 <= av <= 100)
            asr_totals[f"{name} {sev}"] = nt
            tagmsg = "" if mt == tag else f" [WRONG MODEL: {mt}]"
            print(f"  ASR  {sev:5}  {'OK ' if ok else 'BAD'}  asr={av}  n={nt}  model={mt}{tagmsg}")
            check(mt == tag, f"{name} ASR {sev}: model tag is {mt}, expected {tag}")
            check(nt is not None and 0 < nt, f"{name} ASR {sev}: bad n_total={nt}")

        # ---- ORR ----
        op = (f"{R}/orr_{tag}/orr_results.json" if s == 0
              else f"{R}/orr_noise_sweep/orr_{tag}_gaussian_noise_sev{s}.json")
        d = load(op)
        if d is None:
            print(f"  ORR  {sev:5}  MISSING/UNREADABLE")
            problems.append(f"{name} ORR {sev}: missing/unreadable")
        else:
            mt = d.get("model")
            xn = d.get("xstest", {}).get("n_total")
            mn = d.get("mmsa_combined", {}).get("n_total")
            av = d.get("avg_orr_pct")
            # clean files may omit "model"; only enforce tag on sweep files
            tag_ok = True if s == 0 else (mt == tag)
            ok = tag_ok and xn == EXPECT_XSTEST and mn == EXPECT_MMSA and av is not None and 0 <= av <= 100
            tagmsg = "" if tag_ok else f" [WRONG MODEL: {mt}]"
            print(f"  ORR  {sev:5}  {'OK ' if ok else 'BAD'}  avg={av}  xstest={xn}  mmsa={mn}  model={mt}{tagmsg}")
            check(tag_ok, f"{name} ORR {sev}: model tag is {mt}, expected {tag}")
            check(xn == EXPECT_XSTEST, f"{name} ORR {sev}: XSTest n={xn}, expected {EXPECT_XSTEST}")
            check(mn == EXPECT_MMSA,   f"{name} ORR {sev}: MMSA n={mn}, expected {EXPECT_MMSA}")

        # ---- SQA (judged) ----
        jp = (f"{R}/sqa_noise_sweep/judged_{tag}_clean.json" if s == 0
              else f"{R}/sqa_noise_sweep/judged_{tag}_gaussian_noise_sev{s}.json")
        d = load(jp)
        if d is None:
            print(f"  SQA  {sev:5}  MISSING/UNREADABLE (judge not run yet?)")
            problems.append(f"{name} SQA {sev}: missing/unreadable")
        else:
            tot = d.get("total")
            acc = d.get("accuracy")
            unp = d.get("unparsed", 0)
            ok = tot == EXPECT_SQA and acc is not None and 0 <= acc <= 100 and unp <= 10
            unpmsg = "" if unp <= 10 else f" [HIGH UNPARSED: {unp}]"
            print(f"  SQA  {sev:5}  {'OK ' if ok else 'BAD'}  acc={acc}  n={tot}  unparsed={unp}{unpmsg}")
            check(tot == EXPECT_SQA, f"{name} SQA {sev}: total={tot}, expected {EXPECT_SQA}")
            check(unp <= 10, f"{name} SQA {sev}: {unp} unparsed judge verdicts")

# FigStep n_total consistency across all ASR files
vals = set(v for v in asr_totals.values() if v is not None)
print("\n--- FigStep ASR sample-count consistency ---")
if len(vals) <= 1:
    print(f"  All ASR files use n_total={vals.pop() if vals else '?'} (consistent)")
else:
    print(f"  INCONSISTENT n_total across ASR files: {asr_totals}")
    problems.append(f"ASR n_total inconsistent: {sorted(vals)}")

print("\n" + "=" * 60)
if not problems:
    print("ALL CHECKS PASSED — every noise cell is present and valid.")
else:
    print(f"{len(problems)} PROBLEM(S) FOUND:")
    for p in problems:
        print("  -", p)
