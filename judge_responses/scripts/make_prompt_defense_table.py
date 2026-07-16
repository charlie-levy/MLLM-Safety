#!/usr/bin/env python3
"""SIUO prompt-based defense — ASR table (base LLaVA-CoT: none vs +safety vs +blur_safe),
across clean + corruptions. Gridded table: dark outer frame + header rule, light internal
row/column lines. Numbers from the part8 SIUO prompt-defense runs."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/Users/charlielevy/Desktop/REU/judge_responses/figures"

# (corruption, undefended, +safety, +blur_safe)
ROWS = [
    ("clean",      68.3, 59.9, 52.1),
    ("zoom blur",  74.8, 62.3, 53.9),
    ("snow",       71.9, 58.7, 50.9),   # 50.9 = lowest ASR
    ("glass blur", 71.3, 61.7, 56.9),
]
BEST = 50.9
INK="#1A1A1A"; MUTE="#7a7a7a"; FRAME="#2b2b2b"; GRID="#d0d0d0"

B = [0.04, 0.30, 0.52, 0.74, 0.96]        # column boundaries (Corruption wider + 3 number cols)
CX = [B[0]+0.03] + [(B[j]+B[j+1])/2 for j in range(1, 4)]   # label left-anchor + 3 centers
HEADS = ["Corruption", "Undefended", "+ Safety", "+ Blur-safe"]
Y_TOP, Y_BOT = 0.80, 0.14
RH = (Y_TOP - Y_BOT) / (len(ROWS) + 1)
def ry(i): return Y_TOP - (i + 0.5) * RH   # 0 = header
hlines = [Y_TOP - k*RH for k in range(len(ROWS) + 2)]       # every row boundary

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# internal grid (light): row separators + column separators
for y in hlines[2:-1]:
    ax.plot([B[0], B[-1]], [y, y], color=GRID, lw=0.9, zorder=1)
for xb in B[1:-1]:
    ax.plot([xb, xb], [Y_BOT, Y_TOP], color=GRID, lw=0.9, zorder=1)
# frame + header rule (dark)
ax.plot([B[0], B[-1]], [Y_TOP, Y_TOP], color=FRAME, lw=1.8, zorder=2)
ax.plot([B[0], B[-1]], [Y_BOT, Y_BOT], color=FRAME, lw=1.8, zorder=2)
ax.plot([B[0], B[0]], [Y_BOT, Y_TOP], color=FRAME, lw=1.8, zorder=2)
ax.plot([B[-1], B[-1]], [Y_BOT, Y_TOP], color=FRAME, lw=1.8, zorder=2)
ax.plot([B[0], B[-1]], [Y_TOP-RH, Y_TOP-RH], color=FRAME, lw=1.1, zorder=2)

# header
for j, h in enumerate(HEADS):
    ha = "left" if j == 0 else "center"
    ax.text(CX[j], ry(0), h, ha=ha, va="center", fontsize=13.5, fontweight="bold", color=INK)
# rows
for i, (name, und, sf, bs) in enumerate(ROWS, start=1):
    y = ry(i)
    ax.text(CX[0], y, name, ha="left", va="center", fontsize=13, color=INK)
    for j, val in enumerate([und, sf, bs], start=1):
        col = MUTE if j == 1 else INK
        wt = "bold" if (val == BEST) else "normal"
        ax.text(CX[j], y, f"{val:.1f}", ha="center", va="center", fontsize=15, color=col, fontweight=wt)

fig.text(0.5, 0.945, "Prompt-based defense lowers ASR under corruption",
         ha="center", fontsize=17, fontweight="bold", color=INK)
fig.text(0.5, 0.885, "SIUO  ·  Attack Success Rate (ASR, %)  ·  base LLaVA-CoT  ·  lower = safer",
         ha="center", fontsize=11.5, color=MUTE)
fig.subplots_adjust(top=0.86, bottom=0.04, left=0.03, right=0.97)
out = os.path.join(FIG, "prompt_defense_asr_table.png")
fig.savefig(out, dpi=200, facecolor="white"); print("saved", out)
