#!/usr/bin/env python3
"""Part 11 thinking-budget pilot, clean vs zoom_blur (sev2).
R1-Onevision on a 50-sample SIUO subset, forced thinking budgets (0 / 512 / 2048
tokens / natural), judged by the GPT-4o Reasoning/Conclusion harmful-rate judge.
Story: R1's harmful rate is at ceiling regardless of thinking budget, and zoom
blur moves nothing outside pilot noise (n=50 -> 1 sample = 2 pts)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "part11_thinking_budget")
FIG = os.path.join(HERE, "..", "figures")
INK = "#1A1A1A"; MUTE = "#6b6b6b"; GRID = "#ececec"
GRAY = "#9AA0A6"       # clean condition (neutral)
BLUE = "#4C78A8"       # zoom_blur condition

BUDGETS = ["budget0", "budget512", "budget2048", "budgetnatural"]
LABELS = ["0\n(no think)", "512", "2048", "natural"]


def load(path, strip_prefix=""):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["model"].replace(strip_prefix, "")] = (float(row["HR_R"]), float(row["HR_C"]))
    assert set(out) == set(BUDGETS), f"unexpected rows in {path}: {sorted(out)}"
    return out


def main():
    clean = load(os.path.join(RES, "part11_clean_dose_response.csv"))
    zoom = load(os.path.join(RES, "part11_zoomblur_dose_response.csv"), strip_prefix="zoom_")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    x = np.arange(len(BUDGETS)); w = 0.36
    for ax, idx, title in ((axes[0], 0, "Reasoning trace (HR_R)"),
                           (axes[1], 1, "Final answer (HR_C)")):
        cv = [clean[b][idx] for b in BUDGETS]
        zv = [zoom[b][idx] for b in BUDGETS]
        ax.bar(x - w/2, cv, w, color=GRAY, zorder=3, edgecolor="white", linewidth=1, label="clean")
        ax.bar(x + w/2, zv, w, color=BLUE, zorder=3, edgecolor="white", linewidth=1, label="zoom blur")
        for xi, v in zip(x - w/2, cv):
            ax.text(xi, v + 1, "%.0f" % v, ha="center", fontsize=10.5, color=INK)
        for xi, v in zip(x + w/2, zv):
            ax.text(xi, v + 1, "%.0f" % v, ha="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_title(title, fontsize=12.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10.5, color=INK)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(colors=INK, length=0)
    axes[0].set_ylabel("harmful rate (%)", fontsize=11, color=INK)
    axes[0].set_xlabel("thinking budget (tokens)", fontsize=11, color=INK)
    axes[1].set_xlabel("thinking budget (tokens)", fontsize=11, color=INK)
    axes[1].legend(frameon=False, fontsize=10.5, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("Thinking budget doesn't buy safety — and zoom blur can't make R1 much worse",
                 fontsize=14.5, fontweight="bold", color=INK, y=1.0)
    fig.text(0.5, 0.008,
             "R1-Onevision · SIUO 50-sample pilot · forced thinking budgets · GPT-4o R/C judge · "
             "1 sample = 2 pts, so ±4-pt gaps are pilot noise",
             ha="center", fontsize=8.5, color=MUTE)
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.07, right=0.98, wspace=0.08)
    out = os.path.join(FIG, "part11_zoom_dose_response.png")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print("saved", out)

    # console table
    print("\nbudget        clean R/C    zoom R/C     dC (zoom-clean)")
    for b, lab in zip(BUDGETS, ["0 (no think)", "512", "2048", "natural"]):
        print("%-12s  %4.0f/%-4.0f    %4.0f/%-4.0f    %+.0f"
              % (lab, *clean[b], *zoom[b], zoom[b][1] - clean[b][1]))


if __name__ == "__main__":
    main()
