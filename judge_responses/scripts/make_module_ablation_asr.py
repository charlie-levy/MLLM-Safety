#!/usr/bin/env python3
"""SIUO module-ablation ASR under zoom blur (severity 2) — GPT-4o R/C judge, n=167.
Bars: Simple TIS (clean-trained, replaces the old undefended baseline) + TIS applied to
Both / LLM / Vision modules. Two ASR metrics per bar: reasoning-trace ASR and conclusion ASR.
All four scored by the same frozen GPT-4o judge (tis_lora row from summary.csv = Simple TIS)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FIG = "/Users/charlielevy/Desktop/REU/judge_responses/figures"
N = 167
# (label, reasoning-ASR %, n_reason, conclusion-ASR %, n_concl)
BARS = [
    ("Simple TIS\n(clean-trained)", 72.46, 121, 31.14, 52),   # tis_lora, judged 2026-07-13
    ("TIS: Both",                    72.46, 121, 34.73, 58),
    ("TIS: LLM",                     67.07, 112, 31.74, 53),
    ("TIS: Vision",                  50.30, 84,  50.90, 85),
]
INK="#1A1A1A"; MUTE="#6b6b6b"; C_REASON="#E07A5F"; C_CONCL="#3D5A80"

fig, ax = plt.subplots(figsize=(9.2, 5.6))
x = np.arange(len(BARS)); w = 0.38
for i, (_, r, nr, c, nc) in enumerate(BARS):
    ax.bar(x[i]-w/2, r, w, color=C_REASON, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x[i]+w/2, c, w, color=C_CONCL,  edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(x[i]-w/2, r+1.2, f"{r:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.text(x[i]+w/2, c+1.2, f"{c:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.text(x[i]-w/2, 2.0, f"{nr}/{N}", ha="center", va="bottom", fontsize=8.5, color="white")
    ax.text(x[i]+w/2, 2.0, f"{nc}/{N}", ha="center", va="bottom", fontsize=8.5, color="white")

# separate the "baseline defense" (Simple TIS) from the module ablation
ax.axvline(0.5, color="#BBBBBB", ls=(0, (4, 4)), lw=1.1, zorder=1)
ax.text(0.0, 96, "baseline defense", ha="center", fontsize=9.5, color=MUTE, style="italic")
ax.text(2.0, 96, "module ablation", ha="center", fontsize=9.5, color=MUTE, style="italic")

ax.legend(handles=[Patch(facecolor=C_REASON, label="Reasoning-trace ASR"),
                   Patch(facecolor=C_CONCL,  label="Conclusion ASR")],
          frameon=False, loc="upper right", fontsize=10.5)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BARS], fontsize=12, color=INK)
ax.set_ylim(0, 100); ax.set_ylabel("Attack Success Rate (%)  —  lower is safer", fontsize=13, color=INK)
ax.set_title("SIUO: ASR under zoom blur — TIS module ablation", fontsize=17, fontweight="bold", color=INK, pad=14)
ax.grid(axis="y", color="#ececec", lw=1, zorder=0); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(colors=INK)
fig.text(0.5, 0.01, "SIUO (n=167) · zoom blur severity 2 · GPT-4o reasoning/conclusion judge · lower = safer",
         ha="center", fontsize=9.5, color=MUTE)
fig.subplots_adjust(top=0.90, bottom=0.13, left=0.09, right=0.97)
out = os.path.join(FIG, "siuo_module_ablation_asr.png")
fig.savefig(out, dpi=200, facecolor="white"); print("saved", out)
