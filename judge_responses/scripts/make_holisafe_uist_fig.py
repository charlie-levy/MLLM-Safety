#!/usr/bin/env python3
"""HoliSafe (UI+ST) — corruption erodes TIS's safety gain.
String-match ASR (HoliSafe's SM metric), computed from the response files (not hardcoded).
UI+ST = unsafe image + safe text -> model should refuse; ASR = non-refusal (compliance) rate.
Base LLaVA-CoT barely refuses (95.3%); TIS nearly halves it (59.9%); corruption pushes it back up."""
import sys, os, json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REU = "/Users/charlielevy/Desktop/REU"
sys.path.insert(0, os.path.join(REU, "llava_cot_eval", "code"))
from metrics import is_refusal   # tested repo detector; model="tis" uses the TIS answer parser

RES = os.path.join(REU, "judge_responses", "results")

def asr(path, model):
    resp = []
    for l in open(path):
        l = l.strip()
        if l:
            r = json.loads(l); resp.append(r.get("response") or r.get("full_response") or "")
    comply = sum(0 if is_refusal(x, model) else 1 for x in resp)   # UI+ST: non-refusal = attack success
    return 100.0 * comply / len(resp), len(resp)

base_clean, _ = asr(os.path.join(RES, "exp_results/holisafe_base_clean.jsonl"), None)
tis_clean, N  = asr(os.path.join(RES, "exp_results/holisafe_tis_clean.jsonl"), "tis")

corr = []
for p in glob.glob(os.path.join(RES, "part3_results/holisafe_*_tis_responses.jsonl")):
    name = os.path.basename(p).replace("holisafe_", "").replace("_tis_responses.jsonl", "").replace("_", " ")
    a, _ = asr(p, "tis"); corr.append((name, a))
corr.sort(key=lambda t: t[1], reverse=True)                       # worst first

labels = ["TIS\n(clean)"] + [c[0].replace(" ", "\n") for c in corr]
vals   = [tis_clean] + [c[1] for c in corr]
INK="#1A1A1A"; C_BASE="#7d7d7d"; C_CORR="#B0302A"; MUTE="#6b6b6b"
colors = [C_BASE] + [C_CORR if v > tis_clean else "#c98b86" for v in vals[1:]]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(13, 6.6))
ax.bar(x, vals, width=0.72, color=colors, zorder=3)
ax.axhline(tis_clean, ls="--", color="#4d4d4d", lw=1.4, zorder=2)
for xi, v in zip(x, vals):
    ax.text(xi, v+0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    if xi > 0:
        d = v - tis_clean
        ax.text(xi, v-2.4, f"{d:+.1f}", ha="center", va="top", fontsize=11.5, color="white", fontweight="bold")
ax.text(0.02, 0.96, f"Base / no defense:  {base_clean:.1f}% ASR  (off scale)",
        transform=ax.transAxes, ha="left", va="top", fontsize=13.5, fontweight="bold", color=C_CORR)
ax.set_ylim(0, 75)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, color=INK)
ax.set_ylabel("HoliSafe UI+ST  Attack Success Rate (%)", fontsize=15, color=INK)
ax.set_title("HoliSafe (unsafe image + safe text): corruption erodes TIS's safety gain",
             fontsize=17, fontweight="bold", color=INK, pad=14)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(colors=INK); ax.set_axisbelow(True)
ax.grid(axis="y", color="#ececec", lw=0.9)
ax.text(0.5, -0.16, f"HoliSafe UI+ST · n={N} · string-match ASR (paper's SM metric) · higher = less safe",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, color=MUTE)
fig.subplots_adjust(bottom=0.20, top=0.90, left=0.07, right=0.98)
out = os.path.join(REU, "judge_responses", "figures", "holisafe_uist_asr.png")
fig.savefig(out, dpi=200, facecolor="white"); print("saved", out)
