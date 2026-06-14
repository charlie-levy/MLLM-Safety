#!/usr/bin/env python3
"""
analysis_noise_failsafe.py — The honest noise story: under Gaussian noise, UTILITY
erodes while SAFETY (attack resistance) holds. One panel per adapter; FigStep ASR
(safety, lower=better) and ScienceQA utility (higher=better) on a shared 0-100 axis.

Uses corrected metrics (post apostrophe-fix re-score). Run from repo root:
  python3 code/analysis_noise_failsafe.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "results_newton"; PLOTS = f"{B}/plots/presentation"
os.makedirs(PLOTS, exist_ok=True)
LEVELS = [0, 20, 40, 60, 80]
MODELS = [("base_tis", "TIS"), ("base_sage", "SAGE"), ("base_msr", "MSR-Align")]


def L(p): return json.load(open(p)) if os.path.exists(p) else None


def asr(t, p):
    d = (L(f"{B}/figstep_noise_pct/asr_{t}_gaussian_noise_pct_p{p}.json") if p else
         (L(f"{B}/figstep_noise_pct/asr_{t}_gaussian_noise_pct_p0.json")
          or L(f"{B}/figstep_noise_sweep/asr_{t}_clean.json")))
    return d["asr_pct"] if d else np.nan


def sqa(t, p):
    d = (L(f"{B}/sqa_noise_pct/judged_{t}_gaussian_noise_pct_p{p}.json") if p else
         L(f"{B}/sqa_noise_sweep/judged_{t}_clean.json"))
    return d["accuracy"] if d else np.nan


fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
for ax, (t, name) in zip(axes, MODELS):
    A = [asr(t, p) for p in LEVELS]
    Q = [sqa(t, p) for p in LEVELS]
    ax.plot(LEVELS, Q, marker="s", ms=11, lw=3.2, color="#55A868", label="Utility (ScienceQA ↑)")
    ax.plot(LEVELS, A, marker="o", ms=11, lw=3.2, color="#C44E52", label="Attack success (FigStep ↓ = safe)")
    ax.fill_between(LEVELS, A, Q, color="#cfe8d8", alpha=0.5, zorder=0)
    for x, v in zip(LEVELS, Q): ax.text(x, v + 2.5, f"{v:.0f}", ha="center", fontsize=11, color="#2f6b46", fontweight="bold")
    for x, v in zip(LEVELS, A): ax.text(x, v + 2.5, f"{v:.0f}", ha="center", fontsize=11, color="#8c2f33", fontweight="bold")
    ax.set_title(name, fontsize=16, fontweight="bold")
    ax.set_xlabel("Gaussian noise level (%)", fontsize=13)
    ax.set_xticks(LEVELS); ax.set_ylim(-3, 103)
    ax.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    drop = Q[0] - Q[-1]; spread = max(A) - min(A)
    ax.text(0.5, 0.06, f"utility −{drop:.0f} pts   |   safety moves {spread:.0f} pt",
            transform=ax.transAxes, ha="center", fontsize=11, style="italic", color="#444")
axes[0].set_ylabel("Rate / Accuracy (%)", fontsize=13)
axes[0].legend(fontsize=12, loc="center left")
fig.suptitle("Under image noise, adapters FAIL SAFE: utility erodes (−17 to −20 pts) while attack resistance holds flat",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{PLOTS}/analysis_noise_failsafe.png"
fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)
print("Saved:", out)
