#!/usr/bin/env python3
"""Part 10 figures: paper-faithful GPT-4o judging of base LLaVA-CoT on three VLM-safety
benchmarks, clean vs zoom_blur (sev2). Story: image corruption's effect on safety is
SIGNED — it depends on the image's ROLE in the attack.
  MM-SafetyBench (typographic attack: harm is TEXT in the image)  -> blur SAFER  (-11.3)
  VLSBench       (visual leakage: risk only VISIBLE in the image)  -> blur LESS SAFE (+7.0)
  SPA-VL         (natural harmful scene + text; pairwise judge)     -> ~neutral (+4.6 net)
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FIG = "/Users/charlielevy/Desktop/REU/judge_responses/figures"
INK = "#1A1A1A"; MUTE = "#6b6b6b"; GRID = "#ececec"
GRAY = "#9AA0A6"       # clean condition / tie (neutral)
BLUE = "#4C78A8"       # zoom_blur condition
RED = "#C44E52"        # blur MORE unsafe (bad direction)
GREEN = "#4E9A5B"      # blur SAFER (good direction)

# ---- verified numbers (Part 10 summary CSVs) ----
MM_CLEAN, MM_BLUR = 83.33, 72.02          # MM-SafetyBench-Tiny ASR (text-only GPT-4o), n=168
VLS_CLEAN, VLS_BLUR = 81.6, 88.6          # VLSBench ASR = Unsafe (multimodal GPT-4o), n=500
SPA = dict(blur=39.29, tie=26.02, clean=34.69, cons=196, total=265)   # SPA-VL pairwise, GPT-4o
HS_CLEAN, HS_BLUR = 83.61, 87.39          # HoliSafe SI+ST->U (SSU) is_refusal ASR, n=476


# ── FIG 1 (MONEY): signed effect of blur on unsafety, all four benchmarks ────────
def fig_signed():
    rows = [                       # (label, signed effect) — ordered helps -> hurts
        ("MM-SafetyBench\n(attack IN the image: typography)", MM_BLUR - MM_CLEAN),
        ("HoliSafe SI+ST→U\n(safe img+text, harmful combo)", HS_BLUR - HS_CLEAN),
        ("SPA-VL\n(natural scene + text; pairwise net)", SPA["blur"] - SPA["clean"]),
        ("VLSBench\n(risk only VISIBLE in the image)", VLS_BLUR - VLS_CLEAN),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, d) in zip(y, rows):
        c = RED if d > 0 else GREEN
        ax.barh(yi, d, height=0.5, color=c, zorder=3, edgecolor="white", linewidth=1)
        off = 0.5 if d > 0 else -0.5
        ax.text(d + off, yi, ("+%.1f" % d) if d > 0 else ("%.1f" % d), va="center",
                ha="left" if d > 0 else "right", fontsize=15, fontweight="bold", color=INK)
    ax.axvline(0, color="#444", lw=1.4, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11.5, color=INK)
    ax.set_xlim(-15, 11); ax.set_ylim(-0.6, len(rows)-0.05)
    ax.set_xlabel("change in Attack Success Rate under zoom blur (percentage points)", fontsize=11.5, color=INK)
    ax.text(-7.5, len(rows)-0.28, "◄ blur SAFER", ha="center", fontsize=11.5, color=GREEN, fontweight="bold")
    ax.text(5.5, len(rows)-0.28, "blur LESS SAFE ►", ha="center", fontsize=11.5, color=RED, fontweight="bold")
    fig.suptitle("Corruption helps safety only when the image IS the attack",
                 fontsize=15.5, fontweight="bold", color=INK, y=0.975)
    ax.grid(axis="x", color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(colors=INK, length=0)
    fig.text(0.5, 0.015, "base LLaVA-CoT · clean vs zoom_blur (sev2) · 4 VLM-safety benchmarks · MM/VLS/SPA judged by each paper's GPT-4o, HoliSafe by is_refusal",
             ha="center", fontsize=8.5, color=MUTE)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.30, right=0.96)
    out = os.path.join(FIG, "part10_signed_effect.png")
    fig.savefig(out, dpi=200, facecolor="white"); print("saved", out)


# ── FIG 2: the absolute ASR bars behind the two per-response benchmarks ──────────
def fig_asr_bars():
    data = [("MM-SafetyBench\n(text-only judge)", MM_CLEAN, MM_BLUR, 168),
            ("VLSBench\n(multimodal judge)", VLS_CLEAN, VLS_BLUR, 500)]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = np.arange(len(data)); w = 0.34
    for i, (lbl, cl, bl, n) in enumerate(data):
        ax.bar(x[i]-w/2, cl, w, color=GRAY, zorder=3, edgecolor="white", linewidth=1)
        ax.bar(x[i]+w/2, bl, w, color=BLUE, zorder=3, edgecolor="white", linewidth=1)
        for xx, v in ((x[i]-w/2, cl), (x[i]+w/2, bl)):
            ax.text(xx, v+1.2, "%.1f" % v, ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
        d = bl - cl
        ax.annotate(("+%.1f" % d) if d > 0 else ("%.1f" % d),
                    xy=(x[i], max(cl, bl)+3.5), ha="center", fontsize=12.5, fontweight="bold",
                    color=RED if d > 0 else GREEN)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], fontsize=12, color=INK)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Attack Success Rate (%)  —  lower = safer", fontsize=12.5, color=INK)
    ax.set_title("Corruption moves ASR in OPPOSITE directions",
                 fontsize=15, fontweight="bold", color=INK, pad=10)
    ax.legend(handles=[Patch(facecolor=GRAY, label="clean"), Patch(facecolor=BLUE, label="zoom blur (sev2)")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=11, ncol=2)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(colors=INK)
    fig.text(0.5, 0.015, "base LLaVA-CoT · ASR = unsafe/total via each paper's GPT-4o judge (MM-SafetyBench n=168, VLSBench n=500)",
             ha="center", fontsize=9.5, color=MUTE)
    fig.subplots_adjust(top=0.90, bottom=0.24, left=0.10, right=0.97)
    out = os.path.join(FIG, "part10_asr_bars.png")
    fig.savefig(out, dpi=200, facecolor="white"); print("saved", out)


# ── FIG 3: SPA-VL pairwise outcome (diverging stacked bar) ───────────────────────
def fig_spavl():
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    segs = [("blur more harmful", SPA["blur"], RED),
            ("tie", SPA["tie"], GRAY),
            ("clean more harmful", SPA["clean"], GREEN)]
    left = 0
    for name, v, c in segs:
        ax.barh(0, v, left=left, height=0.5, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.text(left + v/2, 0, "%.0f%%" % v, ha="center", va="center", fontsize=13,
                fontweight="bold", color="white")
        ax.text(left + v/2, -0.42, name, ha="center", va="top", fontsize=10.5, color=INK)
        left += v
    ax.set_xlim(0, 100); ax.set_ylim(-0.8, 0.7); ax.axis("off")
    ax.set_title("SPA-VL: does blur make base LLaVA-CoT more harmful? — roughly a coin-flip",
                 fontsize=14.5, fontweight="bold", color=INK, pad=10)
    fig.text(0.5, 0.06, "SPA-VL · pairwise GPT-4o harmlessness, order-swapped + consistency-filtered · "
             "%d/%d consistent · net +%.1f toward more harmful under blur"
             % (SPA["cons"], SPA["total"], SPA["blur"]-SPA["clean"]),
             ha="center", fontsize=8.5, color=MUTE)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.04, right=0.96)
    out = os.path.join(FIG, "part10_spavl_pairwise.png")
    fig.savefig(out, dpi=200, facecolor="white"); print("saved", out)


if __name__ == "__main__":
    fig_signed(); fig_asr_bars(); fig_spavl()
