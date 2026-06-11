#!/usr/bin/env python3
"""
plot_results.py — Generate publication-quality plots from eval results.

Saves all figures to results_newton/plots/.

Usage: python3 code/plot_results.py
"""
import json, os, glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

BASE = "results_newton"
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

COLORS = {
    "base": "#4C72B0",
    "tis":  "#DD8452",
    "sage": "#55A868",
    "msr":  "#C44E52",
}

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_asr(d):
    if "asr_pct" in d:
        return d["asr_pct"]
    return d["asr"]  # older noise-sweep files store percentage directly


# ── 1. FigStep ASR vs Noise Severity ────────────────────────────────────────

def plot_asr_noise():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, label, color in [
        ("base",     "Base",       COLORS["base"]),
        ("base_tis", "Base + TIS", COLORS["tis"]),
        ("base_sage","Base + SAGE", COLORS["sage"]),
        ("base_msr", "Base + MSR", COLORS["msr"]),
    ]:
        clean_p = f"{BASE}/figstep_noise_sweep/asr_{model}_clean.json"
        if not os.path.exists(clean_p):
            continue
        clean_val = get_asr(load(clean_p))

        sevs, asrs = [0], [clean_val]
        for s in [1, 2, 3, 4, 5]:
            p = f"{BASE}/figstep_noise_sweep/asr_{model}_gaussian_noise_sev{s}.json"
            if os.path.exists(p):
                sevs.append(s)
                asrs.append(get_asr(load(p)))

        has_sweep = len(sevs) > 1
        if has_sweep:
            ax.plot(sevs, asrs, marker="o", label=label, color=color,
                    linewidth=2, markersize=6)
        else:
            # Only clean available — draw dashed reference line + annotate
            ax.axhline(clean_val, color=color, linewidth=1.5, linestyle="--",
                       label=f"{label} (clean only)", alpha=0.8)
            ax.plot(0, clean_val, marker="o", color=color, markersize=7, zorder=5)
            ax.annotate(f"{clean_val:.1f}%", xy=(0, clean_val),
                        xytext=(0.2, clean_val + 3), color=color, fontsize=9)

    ax.set_xlabel("Gaussian Noise Severity (0 = clean)")
    ax.set_ylabel("FigStep ASR (%)")
    ax.set_title("FigStep Attack Success Rate vs. Noise Severity")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(-2, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(PLOTS, "asr_noise_sweep.png")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    plt.close(fig)


# ── 2. ORR (Avg) vs Severity — noise and blur ────────────────────────────────

def plot_orr_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    combos = [
        ("gaussian_noise", "orr_noise_sweep", "Gaussian Noise", axes[0]),
        ("gaussian_blur",  "orr_blur_sweep",  "Gaussian Blur",  axes[1]),
    ]

    for noise_type, folder, title, ax in combos:
        for model, label, color in [
            ("base",      "Base",        COLORS["base"]),
            ("base_tis",  "Base + TIS",  COLORS["tis"]),
            ("base_sage", "Base + SAGE", COLORS["sage"]),
            ("base_msr",  "Base + MSR",  COLORS["msr"]),
        ]:
            # clean baseline from orr/ folder
            clean_file = f"{BASE}/orr/orr_{model}.json"
            sevs, orrs = [0], []
            if os.path.exists(clean_file):
                orrs.append(load(clean_file)["avg_orr_pct"])
            else:
                continue

            for s in [1, 2, 3, 4, 5]:
                p = f"{BASE}/{folder}/orr_{model}_{noise_type}_sev{s}.json"
                if os.path.exists(p):
                    sevs.append(s)
                    orrs.append(load(p)["avg_orr_pct"])

            # only draw if we have at least one sweep point beyond clean
            if len(sevs) > 1:
                ax.plot(sevs, orrs, marker="o", label=label, color=color, linewidth=2, markersize=6)

        ax.set_xlabel("Severity (0 = clean)")
        ax.set_title(f"Avg ORR vs. {title} Severity")
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Avg Over-Refusal Rate (%)")
    fig.suptitle("Over-Refusal Rate Under Image Corruptions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTS, "orr_sweep.png")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    plt.close(fig)


# ── 3. Clean Model Comparison Bar Chart ──────────────────────────────────────

def plot_model_comparison():
    """3-panel clean comparison (ORR / ASR / SQA), read from result files.
    Only models with data are shown, so this fills in automatically as SAGE/MSR
    results land."""
    candidates = [
        ("base",      "Base"),
        ("base_tis",  "Base+TIS"),
        ("base_sage", "Base+SAGE"),
        ("base_msr",  "Base+MSR"),
    ]

    labels, xstest, mmsa, avg, asr, sqa = [], [], [], [], [], []
    for tag, name in candidates:
        orr_d = load(f"{BASE}/orr/orr_{tag}.json")
        asr_d = load(f"{BASE}/figstep_noise_sweep/asr_{tag}_clean.json")
        sqa_d = load(f"{BASE}/sqa_noise_sweep/judged_{tag}_clean.json")
        # require at least one metric present to include the model
        if not (orr_d or asr_d or sqa_d):
            continue
        labels.append(name)
        xstest.append(orr_d["xstest"]["orr_pct"]        if orr_d else np.nan)
        mmsa.append(  orr_d["mmsa_combined"]["orr_pct"] if orr_d else np.nan)
        avg.append(   orr_d["avg_orr_pct"]              if orr_d else np.nan)
        asr.append(   get_asr(asr_d)                    if asr_d else np.nan)
        sqa.append(   sqa_d["accuracy"]                 if sqa_d else np.nan)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ── ORR panel: grouped bars colored BY METRIC (clean, distinct hues) ──
    ax = axes[0]
    w = 0.25
    metric_colors = {"XSTest": "#8DA0CB", "MMSA": "#FC8D62", "Avg": "#66C2A5"}
    for off, vals, mlabel in [(-w, xstest, "XSTest"), (0, mmsa, "MMSA"), (w, avg, "Avg")]:
        bars = ax.bar(x + off, vals, w, label=f"{mlabel} ORR", color=metric_colors[mlabel])
        for xi, v in zip(x + off, vals):
            if not np.isnan(v):
                ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Over-Refusal Rate (Clean)")
    ax.set_ylabel("Over-Refusal Rate (%)")
    ax.legend(fontsize=9)

    # ── ASR panel ──
    ax = axes[1]
    bars = ax.bar(x, asr, 0.5, color=[COLORS["base"]] * len(x))
    for xi, v in zip(x, asr):
        if not np.isnan(v):
            ax.text(xi, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("FigStep Attack Success Rate (Clean)")
    ax.set_ylabel("FigStep ASR (%)")

    # ── SQA panel (zoomed, utility differences are small) ──
    ax = axes[2]
    bars = ax.bar(x, sqa, 0.5, color=[COLORS["sage"]] * len(x))
    for xi, v in zip(x, sqa):
        if not np.isnan(v):
            ax.text(xi, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("ScienceQA Utility (Clean)")
    ax.set_ylabel("SQA Judged Accuracy (%)")
    ax.set_ylim(80, 100)  # zoom: all models cluster 88-92%

    for ax in axes[:2]:
        ax.set_ylim(0, 105)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Safety & Utility: Clean Baseline Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_comparison_clean.png")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    plt.close(fig)


# ── 4. ASR + Utility vs Severity (one plot per corruption type) ──────────────

def _plot_asr_utility_sweep(noise_type, fig_folder, sqa_folder, fig_title, outname):
    """
    Two side-by-side subplots: left = FigStep ASR, right = SQA Utility.
    Each has a single clean y-axis. One line per model (only models with data
    for this corruption type are drawn).
    """
    fig, (ax_asr, ax_sqa) = plt.subplots(1, 2, figsize=(11, 4.5))

    models = [
        ("base",      "Base",        COLORS["base"], "o"),
        ("base_tis",  "Base + TIS",  COLORS["tis"],  "s"),
        ("base_sage", "Base + SAGE", COLORS["sage"], "^"),
        ("base_msr",  "Base + MSR",  COLORS["msr"],  "D"),
    ]

    for model_tag, label, color, marker in models:
        clean_asr = load(f"{BASE}/figstep_noise_sweep/asr_{model_tag}_clean.json")
        # clean SQA is shared across corruption types and always lives in the
        # noise-sweep folder (there is no separate "clean blur" run)
        clean_sqa = load(f"{BASE}/sqa_noise_sweep/judged_{model_tag}_clean.json")

        asr_sevs, asr_vals = [], []
        sqa_sevs, sqa_vals = [], []

        if clean_asr:
            asr_sevs.append(0); asr_vals.append(get_asr(clean_asr))
        if clean_sqa:
            sqa_sevs.append(0); sqa_vals.append(clean_sqa["accuracy"])

        for s in [1, 2, 3, 4, 5]:
            ad = load(f"{BASE}/{fig_folder}/asr_{model_tag}_{noise_type}_sev{s}.json")
            sd = load(f"{BASE}/{sqa_folder}/judged_{model_tag}_{noise_type}_sev{s}.json")
            if ad:
                asr_sevs.append(s); asr_vals.append(get_asr(ad))
            if sd:
                sqa_sevs.append(s); sqa_vals.append(sd["accuracy"])

        if asr_sevs:
            ax_asr.plot(asr_sevs, asr_vals, color=color, marker=marker,
                        linewidth=2, markersize=7, label=label)
        if sqa_sevs:
            ax_sqa.plot(sqa_sevs, sqa_vals, color=color, marker=marker,
                        linewidth=2, markersize=7, label=label)

    for ax, ylabel, ylim in [
        (ax_asr, "FigStep ASR (%)",          (0, 100)),
        (ax_sqa, "SQA Judged Accuracy (%)",  (85, 100)),
    ]:
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    ax_asr.set_title("FigStep ASR vs. Severity")
    ax_sqa.set_title("SQA Utility vs. Severity")
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTS, outname)
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    plt.close(fig)


def plot_asr_vs_utility():
    _plot_asr_utility_sweep(
        "gaussian_noise", "figstep_noise_sweep", "sqa_noise_sweep",
        "Gaussian Noise Corruption",
        "asr_utility_noise.png",
    )
    _plot_asr_utility_sweep(
        "gaussian_blur", "figstep_blur_sweep", "sqa_blur_sweep",
        "Gaussian Blur Corruption",
        "asr_utility_blur.png",
    )


if __name__ == "__main__":
    plot_asr_noise()
    plot_orr_sweep()
    plot_model_comparison()
    plot_asr_vs_utility()
    print("\nAll plots saved to", PLOTS)
