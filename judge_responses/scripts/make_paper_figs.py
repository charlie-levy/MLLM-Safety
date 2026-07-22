#!/usr/bin/env python3
"""Camera-ready figures for the WACV draft -> REU_WACV/figures/*.pdf (vector).
Single-column WACV width ~3.25in; fonts sized for \\linewidth inclusion.

fig 1  decoupling.pdf   scatter of (delta utility, delta HR_C) per corruption for the
                        SAME TIS-tuned LLaVA-CoT: safety degrades even when utility
                        doesn't (all 10 points above y=0; 3 at x>=0).
fig 2  signed_effect.pdf  signed ASR change under zoom blur across benchmarks that
                        differ in where the harm lives (base LLaVA-CoT).
Data provenance: sqa_utility_plot.png / corruptions_bring_asr_back_up.png (TIS,
verified deck numbers) and make_part10_figs.py (part10 summary CSVs)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "REU_WACV", "figures"))
os.makedirs(OUT, exist_ok=True)
INK = "#1A1A1A"; MUTE = "#6b6b6b"; GRID = "#e5e5e5"
BLUE = "#4C78A8"; RED = "#C44E52"; GREEN = "#4E9A5B"; GRAY = "#9AA0A6"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "pdf.fonttype": 42})

# (corruption, delta SQA utility pp, delta SIUO ASR pp) — TIS LLaVA-CoT, clean = 90.4 / 25.1
DECOUPLE = [
    ("glass blur",   -8.4, +5.4),
    ("defocus blur", -6.8, +3.6),
    ("motion blur",  -4.8, +1.3),
    ("zoom blur",    -4.0, +8.4),
    ("contrast",     -3.6, +4.2),
    ("snow",         -2.0, +4.2),
    ("elastic",      -1.6, +1.9),
    ("JPEG",          0.0, +4.2),
    ("frost",        +0.8, +4.2),
    ("fog",          +1.6, +2.4),
]

# label placement offsets to avoid collisions (x_pt, y_pt)
NUDGE = {"snow": (4, -3), "contrast": (4, 3), "frost": (4, 2), "JPEG": (4, -8),
         "fog": (-2, -9), "elastic": (4, -3), "motion blur": (4, -3),
         "zoom blur": (4, -2), "glass blur": (4, -2), "defocus blur": (4, -2)}


def fig_decoupling():
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    for name, du, da in DECOUPLE:
        ax.scatter(du, da, s=26, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
        dx, dy = NUDGE.get(name, (4, -2))
        ax.annotate(name, (du, da), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.5, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-10.5, 3.6); ax.set_ylim(-1.2, 9.6)
    ax.set_xlabel(r"$\Delta$ ScienceQA accuracy (pp)  $\leftarrow$ utility lost")
    ax.set_ylabel(r"$\Delta$ SIUO HR$_\mathrm{C}$ (pp)  less safe $\rightarrow$")
    ax.text(2.9, 0.4, "no utility\nloss", fontsize=6.5, color=MUTE, ha="right", va="bottom")
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, "decoupling.pdf")
    fig.savefig(out); print("saved", out)
    fig.savefig(out[:-4] + ".png", dpi=600)  # high-res raster for slides/poster


# (benchmark, where the harm lives, delta ASR under zoom blur) — base LLaVA-CoT
SIGNED = [
    ("MM-SafetyBench",  "attack rendered in image",   -11.3),
    ("SPA-VL",          "natural scene + text",        +4.6),
    ("HoliSafe",        "image+text jointly unsafe",   +3.8),
    ("VLSBench",        "risk visible only in image",  +7.0),
]


def fig_signed():
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    y = list(range(len(SIGNED) - 1, -1, -1))
    for yi, (name, sub, d) in zip(y, SIGNED):
        c = RED if d > 0 else GREEN
        ax.barh(yi, d, height=0.55, color=c, zorder=3)
        if d > 0:
            ax.text(d + 0.4, yi, f"{d:+.1f}", va="center", ha="left",
                    fontsize=7.5, fontweight="bold")
        else:
            ax.text(d + 0.5, yi, f"{d:+.1f}", va="center", ha="left",
                    fontsize=7.5, fontweight="bold", color="white")
    ax.axvline(0, color=INK, lw=0.9, zorder=4)
    ax.set_xlim(-13, 10.8); ax.set_ylim(-0.55, len(SIGNED) - 0.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}\n{sub}" for name, sub, _ in SIGNED], fontsize=6.8)
    ax.set_xlabel(r"$\Delta$ ASR under zoom blur (pp)")
    ax.text(-6.5, len(SIGNED) - 0.42, r"safer $\blacktriangleleft$", fontsize=7,
            color=GREEN, ha="center", fontweight="bold")
    ax.text(4.5, len(SIGNED) - 0.42, r"$\blacktriangleright$ less safe", fontsize=7,
            color=RED, ha="center", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, "signed_effect.pdf")
    fig.savefig(out); print("saved", out)
    fig.savefig(out[:-4] + ".png", dpi=600)  # high-res raster for slides/poster


if __name__ == "__main__":
    fig_decoupling()
    fig_signed()
