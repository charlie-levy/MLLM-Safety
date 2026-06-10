#!/usr/bin/env python
"""
plot_orr_vs_asr.py — ORR vs ASR scatter showing safety/over-refusal trade-off.

X-axis: ORR (%) — over-refusal, higher = worse utility
Y-axis: ASR (%) — attack success, higher = less safe
Ideal = bottom-left corner (low ORR, low ASR)

Each point = one condition (clean, noise/blur sev 1-5).
Reads from results_newton/ — run after scp of ORR noise sweep results.
"""
import json, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RES      = os.path.join(BASE_DIR, "results_newton")

def load_asr(model, noise_type, sev):
    if sev == 0:
        path = os.path.join(RES, "figstep_noise_sweep", "asr_%s_clean.json" % model)
    elif noise_type == "gaussian_noise":
        path = os.path.join(RES, "figstep_noise_sweep", "asr_%s_gaussian_noise_sev%d.json" % (model, sev))
    else:
        path = os.path.join(RES, "figstep_blur_sweep",  "asr_%s_gaussian_blur_sev%d.json"  % (model, sev))
    with open(path) as f: d = json.load(f)
    return d.get("asr_pct") or d.get("asr")

def load_orr_clean(model):
    with open(os.path.join(RES, "orr", "orr_%s.json" % model)) as f:
        d = json.load(f)
    return d["avg_orr_pct"]

def load_orr_noise(model, noise_type, sev):
    fname = "orr_%s_%s_sev%d.json" % (model, noise_type, sev)
    with open(os.path.join(RES, "orr_noise_sweep", fname)) as f:
        d = json.load(f)
    return d["avg_orr_pct"]

SEVS = [0, 1, 2, 3, 4, 5]
STYLE = {
    ("base",     "gaussian_noise"): dict(color="#1565C0", marker="o", label="Base + Noise"),
    ("base",     "gaussian_blur"):  dict(color="#42A5F5", marker="s", label="Base + Blur"),
    ("base_tis", "gaussian_noise"): dict(color="#B71C1C", marker="o", label="TIS + Noise"),
    ("base_tis", "gaussian_blur"):  dict(color="#EF5350", marker="s", label="TIS + Blur"),
}

fig, ax = plt.subplots(figsize=(9, 7))
ax.set_facecolor("#FAFAFA")

for model in ["base", "base_tis"]:
    for noise_type in ["gaussian_noise", "gaussian_blur"]:
        st = STYLE[(model, noise_type)]
        orr_vals, asr_vals, sev_vals = [], [], []

        for sev in SEVS:
            try:
                asr = load_asr(model, noise_type, sev)
                orr = load_orr_clean(model) if sev == 0 else load_orr_noise(model, noise_type, sev)
                orr_vals.append(orr); asr_vals.append(asr); sev_vals.append(sev)
            except FileNotFoundError:
                print("  MISSING: %s %s sev%d" % (model, noise_type, sev))

        if not orr_vals: continue

        ax.plot(orr_vals, asr_vals, color=st["color"], marker=st["marker"],
                ls="-", lw=1.5, markersize=9, label=st["label"], alpha=0.85)

        for sev, orr, asr in zip(sev_vals, orr_vals, asr_vals):
            lbl = "clean" if sev == 0 else "s%d" % sev
            ax.annotate(lbl, xy=(orr, asr), xytext=(5, 4),
                        textcoords="offset points",
                        fontsize=8, color=st["color"])

# Ideal corner
ax.annotate("★ ideal\n(low ORR,\nlow ASR)", xy=(5, 5), fontsize=9,
            color="#2E7D32", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", alpha=0.7))

ax.set_xlabel("Avg ORR  (%)  →  higher = more over-refusal", fontsize=11)
ax.set_ylabel("FigStep ASR  (%)  →  higher = less safe", fontsize=11)
ax.set_title("ORR vs. ASR Trade-off Under Image Corruption\n(LLaVA-CoT Base vs. Think-in-Safety)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.25, linestyle="--")
ax.set_xlim(0, 100); ax.set_ylim(0, 85)
ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

plt.tight_layout()
out = os.path.join(RES, "plot_orr_vs_asr.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
plt.show()
