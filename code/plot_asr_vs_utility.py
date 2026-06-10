#!/usr/bin/env python
"""
plot_asr_vs_utility.py — ASR (FigStep) vs ScienceQA Utility trade-off.
Each point = one severity level (0=clean, 1-5).
Two panels: Gaussian Noise | Gaussian Blur.
"""
import json, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RES = os.path.join(BASE_DIR, "results_newton")

def load_asr(model, noise_type, sev):
    if sev == 0:
        path = os.path.join(RES, "figstep_noise_sweep", "asr_%s_clean.json" % model)
    elif noise_type == "gaussian_noise":
        path = os.path.join(RES, "figstep_noise_sweep", "asr_%s_gaussian_noise_sev%d.json" % (model, sev))
    else:
        path = os.path.join(RES, "figstep_blur_sweep",  "asr_%s_gaussian_blur_sev%d.json"  % (model, sev))
    with open(path) as f: d = json.load(f)
    return d.get("asr_pct") or d.get("asr")

def load_sqa(model, noise_type, sev):
    if sev == 0:
        path = os.path.join(RES, "sqa_noise_sweep", "acc_%s_clean.json" % model)
    elif noise_type == "gaussian_noise":
        path = os.path.join(RES, "sqa_noise_sweep", "acc_%s_gaussian_noise_sev%d.json" % (model, sev))
    else:
        path = os.path.join(RES, "sqa_blur_sweep",  "acc_%s_gaussian_blur_sev%d.json"  % (model, sev))
    with open(path) as f: d = json.load(f)
    return d["accuracy"]

SEVS = [0, 1, 2, 3, 4, 5]

STYLE = {
    ("base",     "gaussian_noise"): dict(color="#1565C0", marker="o", ls="-",  label="Base + Noise",    zorder=3),
    ("base",     "gaussian_blur"):  dict(color="#42A5F5", marker="s", ls="--", label="Base + Blur",     zorder=3),
    ("base_tis", "gaussian_noise"): dict(color="#B71C1C", marker="o", ls="-",  label="TIS + Noise",     zorder=4),
    ("base_tis", "gaussian_blur"):  dict(color="#EF5350", marker="s", ls="--", label="TIS + Blur",      zorder=4),
}

def build_curve(model, noise_type):
    pts = []
    for sev in SEVS:
        try:
            pts.append((sev, load_asr(model, noise_type, sev), load_sqa(model, noise_type, sev)))
        except FileNotFoundError:
            print("  MISSING: %s %s sev%d" % (model, noise_type, sev))
    return pts

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle("Safety–Utility Trade-off Under Image Corruption\n(LLaVA-CoT Base vs. Think-in-Safety)",
             fontsize=14, fontweight="bold", y=1.01)

for ax, noise_type in zip(axes, ["gaussian_noise", "gaussian_blur"]):
    for model in ["base", "base_tis"]:
        st  = STYLE[(model, noise_type)]
        pts = build_curve(model, noise_type)
        if not pts: continue
        sevs, asrs, sqas = zip(*pts)

        ax.plot(sqas, asrs, color=st["color"], marker=st["marker"],
                ls=st["ls"], lw=2, markersize=8, label=st["label"], zorder=st["zorder"])

        for sev, sqa, asr in zip(sevs, sqas, asrs):
            lbl  = "clean" if sev == 0 else "s%d" % sev
            off  = (6, 5) if model == "base" else (-28, -14)
            ax.annotate(lbl, xy=(sqa, asr), xytext=off,
                        textcoords="offset points",
                        fontsize=8.5, fontweight="bold" if sev == 0 else "normal",
                        color=st["color"])

    noise_label = "Gaussian Noise" if noise_type == "gaussian_noise" else "Gaussian Blur"
    ax.set_title(noise_label, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("ScienceQA Accuracy  (%)  →  higher = more useful", fontsize=10)
    ax.set_ylabel("FigStep ASR  (%)  →  lower = safer", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(25, 95)
    ax.set_ylim(0, 82)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

    # Ideal corner annotation
    ax.annotate("← ideal", xy=(30, 5), fontsize=9, color="green",
                fontstyle="italic", alpha=0.7)

plt.tight_layout()
out = os.path.join(RES, "plot_asr_vs_utility.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
plt.show()
