#!/usr/bin/env python3
"""Dose-response figure: SIUO harmful-rate vs zoom-blur severity (0=clean..5) for
LLaVA-CoT and Qwen2.5-VL. Data = part12_dose_response.csv (GPT-4o R/C judge, n=167
per cell, re-judged clean after the out-of-credits poisoning was purged).
Two panels: reasoning-trace HR_R and final-answer HR_C. -> REU_WACV/figures/."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "REU_WACV", "figures"))
os.makedirs(OUT, exist_ok=True)
INK = "#1A1A1A"; MUTE = "#6b6b6b"; GRID = "#e5e5e5"
BLUE = "#4C78A8"; ORANGE = "#D1812C"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "pdf.fonttype": 42})

SEV = [0, 1, 2, 3, 4, 5]
# from part12_dose_response.csv (clean/verified)
HR_R = {"LLaVA-CoT":   [68.26, 77.25, 80.84, 79.64, 78.44, 80.84],
        "Qwen2.5-VL":  [69.46, 74.85, 74.25, 73.05, 74.85, 72.46]}
HR_C = {"LLaVA-CoT":   [68.26, 71.26, 73.05, 74.25, 73.65, 70.66],
        "Qwen2.5-VL":  [70.06, 74.25, 73.05, 73.05, 74.25, 73.05]}
COL = {"LLaVA-CoT": BLUE, "Qwen2.5-VL": ORANGE}


def panel(ax, data, title):
    for model, ys in data.items():
        ax.plot(SEV, ys, "-o", color=COL[model], lw=1.6, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.6, label=model, zorder=3)
    ax.axvspan(-0.35, 0.35, color=GRID, zorder=0)  # mark the clean cell
    ax.text(0, ax.get_ylim()[0], "", fontsize=6)
    ax.set_xticks(SEV)
    ax.set_xticklabels(["clean", "1", "2", "3", "4", "5"])
    ax.set_xlabel("zoom-blur severity")
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.5, 5.3)


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    panel(a1, HR_R, "Reasoning trace (HR$_\\mathrm{R}$)")
    panel(a2, HR_C, "Final answer (HR$_\\mathrm{C}$)")
    a1.set_ylabel("harmful rate (%)")
    a1.set_ylim(64, 84)
    a1.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "dose_response." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else None); print("saved", p)


if __name__ == "__main__":
    main()
