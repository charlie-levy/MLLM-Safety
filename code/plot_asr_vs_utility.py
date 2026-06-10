#!/usr/bin/env python
"""
plot_asr_vs_utility.py — ASR vs ScienceQA Utility trade-off curves.

Reads from results_newton/ (local copies of Newton results).
Produces two plots:
  1. Gaussian Noise: ASR vs SQA Accuracy at sev 0-5 (Base + TIS)
  2. Gaussian Blur:  ASR vs SQA Accuracy at sev 0-5 (Base + TIS)

Each point = one severity level. Arrows show direction of increasing severity.
"""
import json, os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RES = os.path.join(BASE_DIR, "results_newton")

def load_asr(model, noise_type, sev):
    if sev == 0:
        fname = "asr_%s_clean.json" % model
        path = os.path.join(RES, "figstep_noise_sweep", fname)
    elif noise_type == "gaussian_noise":
        fname = "asr_%s_gaussian_noise_sev%d.json" % (model, sev)
        path = os.path.join(RES, "figstep_noise_sweep", fname)
    else:
        fname = "asr_%s_gaussian_blur_sev%d.json" % (model, sev)
        path = os.path.join(RES, "figstep_blur_sweep", fname)

    with open(path) as f:
        d = json.load(f)
    return d.get("asr_pct") or d.get("asr")

def load_sqa(model, noise_type, sev):
    if sev == 0:
        fname = "acc_%s_clean.json" % model
        path = os.path.join(RES, "sqa_noise_sweep", fname)
    elif noise_type == "gaussian_noise":
        fname = "acc_%s_gaussian_noise_sev%d.json" % (model, sev)
        path = os.path.join(RES, "sqa_noise_sweep", fname)
    else:
        fname = "acc_%s_gaussian_blur_sev%d.json" % (model, sev)
        path = os.path.join(RES, "sqa_blur_sweep", fname)

    with open(path) as f:
        d = json.load(f)
    return d["accuracy"]

SEVERITIES = [0, 1, 2, 3, 4, 5]
MODELS = ["base", "base_tis"]
NOISE_TYPES = ["gaussian_noise", "gaussian_blur"]

STYLE = {
    ("base",     "gaussian_noise"): dict(color="#2196F3", marker="o", ls="-",  label="Base + Noise"),
    ("base",     "gaussian_blur"):  dict(color="#2196F3", marker="s", ls="--", label="Base + Blur"),
    ("base_tis", "gaussian_noise"): dict(color="#F44336", marker="o", ls="-",  label="TIS + Noise"),
    ("base_tis", "gaussian_blur"):  dict(color="#F44336", marker="s", ls="--", label="TIS + Blur"),
}

def build_curve(model, noise_type):
    asr_vals, sqa_vals = [], []
    for sev in SEVERITIES:
        try:
            asr_vals.append(load_asr(model, noise_type, sev))
            sqa_vals.append(load_sqa(model, noise_type, sev))
        except FileNotFoundError:
            print("  MISSING: %s / %s / sev%d" % (model, noise_type, sev))
            asr_vals.append(None)
            sqa_vals.append(None)
    return asr_vals, sqa_vals

def plot_combined():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, noise_type in zip(axes, NOISE_TYPES):
        for model in MODELS:
            style = STYLE[(model, noise_type)]
            asr_vals, sqa_vals = build_curve(model, noise_type)

            # Filter out missing points
            valid = [(s, a, u) for s, a, u in zip(SEVERITIES, asr_vals, sqa_vals)
                     if a is not None and u is not None]
            if not valid:
                continue
            sevs, asrs, sqas = zip(*valid)

            ax.plot(sqas, asrs,
                    color=style["color"], marker=style["marker"],
                    ls=style["ls"], linewidth=2, markersize=7,
                    label=style["label"])

            # Annotate severity labels
            for sev, sqa, asr in zip(sevs, sqas, asrs):
                label = "clean" if sev == 0 else "s%d" % sev
                ax.annotate(label, (sqa, asr),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=7, color=style["color"])

        noise_label = "Gaussian Noise" if noise_type == "gaussian_noise" else "Gaussian Blur"
        ax.set_title("ASR vs Utility — %s" % noise_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("ScienceQA Accuracy (%) ↑", fontsize=11)
        ax.set_ylabel("FigStep ASR (%) ↓", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(20, 95)
        ax.set_ylim(0, 80)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "results_newton", "plot_asr_vs_utility.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved: %s" % out)
    plt.show()

if __name__ == "__main__":
    plot_combined()
