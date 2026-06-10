#!/usr/bin/env python
"""
plot_orr_noise_sweep.py — ORR vs severity under noise/blur (Base + TIS).
Reads from results_newton/orr_noise_sweep/ after scp from Newton.
Two panels: Gaussian Noise | Gaussian Blur.
"""
import json, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RES      = os.path.join(BASE_DIR, "results_newton", "orr_noise_sweep")

def load_orr(model, noise_type, sev):
    fname = "orr_%s_%s_sev%d.json" % (model, noise_type, sev)
    with open(os.path.join(RES, fname)) as f:
        d = json.load(f)
    xs  = d["xstest"]["orr_pct"]
    mms = d["mmsa_combined"]["orr_pct"]
    avg = d["avg_orr_pct"]
    return xs, mms, avg

def load_orr_clean(model):
    path = os.path.join(BASE_DIR, "results_newton", "orr",
                        "orr_%s.json" % model)
    with open(path) as f:
        d = json.load(f)
    return d["xstest"]["orr_pct"], d["mmsa_combined"]["orr_pct"], d["avg_orr_pct"]

SEVS = [1, 2, 3, 4, 5]
STYLE = {
    ("base",     "xstest"):        dict(color="#1565C0", marker="o", ls="-",  label="Base – XSTest"),
    ("base",     "mmsa_combined"): dict(color="#1565C0", marker="s", ls="--", label="Base – MMSA"),
    ("base_tis", "xstest"):        dict(color="#B71C1C", marker="o", ls="-",  label="TIS – XSTest"),
    ("base_tis", "mmsa_combined"): dict(color="#EF5350", marker="s", ls="--", label="TIS – MMSA"),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle("Over-Refusal Rate (ORR) vs. Image Corruption Severity\n(LLaVA-CoT Base vs. Think-in-Safety)",
             fontsize=14, fontweight="bold", y=1.01)

for ax, noise_type in zip(axes, ["gaussian_noise", "gaussian_blur"]):
    for model in ["base", "base_tis"]:
        # Clean baseline at sev=0
        try:
            xs0, mms0, _ = load_orr_clean(model)
            clean_xs  = xs0
            clean_mms = mms0
        except FileNotFoundError:
            clean_xs = clean_mms = None

        xs_vals, mms_vals = [], []
        sev_axis = []

        for sev in SEVS:
            try:
                xs, mms, _ = load_orr(model, noise_type, sev)
                xs_vals.append(xs); mms_vals.append(mms); sev_axis.append(sev)
            except FileNotFoundError:
                print("  MISSING: %s %s sev%d" % (model, noise_type, sev))

        full_sevs = ([0] + sev_axis) if clean_xs is not None else sev_axis
        full_xs   = ([clean_xs]  + xs_vals)  if clean_xs  is not None else xs_vals
        full_mms  = ([clean_mms] + mms_vals) if clean_mms is not None else mms_vals

        st_xs  = STYLE[(model, "xstest")]
        st_mms = STYLE[(model, "mmsa_combined")]

        ax.plot(full_sevs, full_xs,  color=st_xs["color"],  marker=st_xs["marker"],
                ls=st_xs["ls"],  lw=2, markersize=7, label=st_xs["label"])
        ax.plot(full_sevs, full_mms, color=st_mms["color"], marker=st_mms["marker"],
                ls=st_mms["ls"], lw=2, markersize=7, label=st_mms["label"])

    noise_label = "Gaussian Noise" if noise_type == "gaussian_noise" else "Gaussian Blur"
    ax.set_title(noise_label, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Severity  (0 = clean)", fontsize=10)
    ax.set_ylabel("Over-Refusal Rate ORR  (%)  ↓  lower = better", fontsize=10)
    ax.set_xticks([0,1,2,3,4,5])
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

plt.tight_layout()
out = os.path.join(BASE_DIR, "results_newton", "plot_orr_noise_sweep.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
plt.show()
