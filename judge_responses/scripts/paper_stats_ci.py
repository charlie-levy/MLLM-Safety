#!/usr/bin/env python3
"""Reproduces the confidence intervals and the perception-failure / decoupling
numbers cited in sec/4_results.tex. Login-node safe (stdlib only).

Reported:
  - decoupling: mean ASR gain across the 10 corruptions + 95% CI (t & bootstrap),
    and Pearson r between utility-change and ASR-change (the decoupling test).
  - zoom blur: mean HR_C shift across the 6 configs + 95% CI.
  - perception-failure rate by model x condition (from part4 master_responses.csv).
"""
import os, math, random, csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
random.seed(20260720)
TCRIT = {6: 2.571, 10: 2.262}  # two-sided 95%, df=n-1


def tci(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n); t = TCRIT[n]
    return m, m - t * se, m + t * se


def boot_ci(xs, B=20000):
    n = len(xs); ms = []
    for _ in range(B):
        ms.append(sum(xs[random.randrange(n)] for _ in range(n)) / n)
    ms.sort()
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def pearson(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return cov / (sa * sb)


# --- decoupling: (delta utility, delta ASR) for TIS LLaVA-CoT, 10 corruptions ---
DUTIL = [-8.4, -6.8, -4.8, -4.0, -3.6, -2.0, -1.6, 0.0, 0.8, 1.6]
DASR = [5.4, 3.6, 1.3, 8.4, 4.2, 4.2, 1.9, 4.2, 4.2, 2.4]
m, lo, hi = tci(DASR); blo, bhi = boot_ci(DASR)
print("DECOUPLING  mean ASR gain = %.2f  t95%%=[%.2f,%.2f]  boot95%%=[%.2f,%.2f]  all>0=%s"
      % (m, lo, hi, blo, bhi, all(x > 0 for x in DASR)))
print("            Pearson r(util-change, ASR-change) = %.3f" % pearson(DUTIL, DASR))

# --- zoom blur HR_C shift across the 6 configs (Table 1) ---
CLEAN = [60.5, 68.3, 70.1, 83.2, 83.8, 83.8]
ZOOM = [69.5, 74.8, 74.2, 86.8, 86.8, 88.6]
DZ = [z - c for z, c in zip(ZOOM, CLEAN)]
m, lo, hi = tci(DZ); blo, bhi = boot_ci(DZ)
print("ZOOM-BLUR   mean HR_C shift = %.2f  t95%%=[%.2f,%.2f]  boot95%%=[%.2f,%.2f]  all>0=%s"
      % (m, lo, hi, blo, bhi, all(x > 0 for x in DZ)))

# --- perception-failure rate by model x condition (part4) ---
csv_path = os.path.join(REPO, "analysis", "part4_exploratory", "outputs", "master_responses.csv")
if os.path.exists(csv_path):
    agg = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(csv_path)):
        pf = r["perception_failure"].strip().lower() in ("true", "1")
        agg[(r["model"], r["condition"])][0] += 1 if pf else 0
        agg[(r["model"], r["condition"])][1] += 1
    print("PERCEPTION-FAILURE rate (%, model x condition):")
    conds = ["clean", "glass_blur", "snow", "zoom_blur"]
    for mdl in sorted({k[0] for k in agg}):
        cells = [f"{c}={100*agg[(mdl,c)][0]/agg[(mdl,c)][1]:.1f}" for c in conds if (mdl, c) in agg]
        print("   %-22s %s" % (mdl, "  ".join(cells)))
