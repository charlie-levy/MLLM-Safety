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
            ("base",     "Base",       COLORS["base"]),
            ("base_tis", "Base + TIS", COLORS["tis"]),
        ]:
            # clean from orr/ folder
            clean_file = f"{BASE}/orr/orr_{model}.json" if model == "base" else f"{BASE}/orr/orr_{model.replace('base_','orr_base_')}.json"
            # fix path
            if model == "base":
                clean_file = f"{BASE}/orr/orr_base.json"
            elif model == "base_tis":
                clean_file = f"{BASE}/orr/orr_base_tis.json"

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
    models = ["Base", "Base+TIS", "Base+SAGE"]
    colors = [COLORS["base"], COLORS["tis"], COLORS["sage"]]

    xstest = [26.8, 74.8, 36.0]
    mmsa   = [59.6, 89.7, 65.0]
    avg    = [43.2, 82.3, 50.5]
    asr    = [70.4, 13.8,  0.4]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(models))
    w = 0.25

    # ORR subplot
    ax = axes[0]
    ax.bar(x - w, xstest, w, label="XSTest ORR",  color=[c + "bb" for c in ["#4C72B0", "#DD8452", "#55A868"]])
    ax.bar(x,     mmsa,   w, label="MMSA ORR",    color=[c + "99" for c in ["#4C72B0", "#DD8452", "#55A868"]])
    ax.bar(x + w, avg,    w, label="Avg ORR",     color=colors)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Over-Refusal Rate (%)")
    ax.set_title("Over-Refusal Rate (Clean)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bars, vals in zip([x - w, x, x + w], [xstest, mmsa, avg]):
        for xi, v in zip(bars, vals):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # ASR subplot
    ax = axes[1]
    bars = ax.bar(x, asr, 0.5, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("FigStep ASR (%)")
    ax.set_title("FigStep Attack Success Rate (Clean)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, asr):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Safety Metrics: Clean Baseline Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_comparison_clean.png")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    plt.close(fig)


# ── 4. ASR vs SQA Utility scatter ────────────────────────────────────────────

def plot_asr_vs_utility():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # shared clean point (same for both noise and blur)
    clean_asr_d = load(f"{BASE}/figstep_noise_sweep/asr_base_clean.json")
    clean_sqa_d = load(f"{BASE}/sqa_noise_sweep/judged_base_clean.json")

    configs = [
        ("base", "gaussian_noise", "figstep_noise_sweep", "sqa_noise_sweep",
         "Base + Noise", COLORS["base"], "o", "#4C72B0"),
        ("base", "gaussian_blur",  "figstep_blur_sweep",  "sqa_blur_sweep",
         "Base + Blur",  "#E05050",     "s", "#E05050"),
    ]

    last_sc = None
    for model, noise, fig_folder, sqa_folder, label, color, marker, ec in configs:
        pts = []
        for sev in [1, 2, 3, 4, 5]:
            asr_d = load(f"{BASE}/{fig_folder}/asr_{model}_{noise}_sev{sev}.json")
            sqa_d = load(f"{BASE}/{sqa_folder}/judged_{model}_{noise}_sev{sev}.json")
            if asr_d and sqa_d:
                pts.append((sqa_d["accuracy"], get_asr(asr_d), sev))
        if not pts:
            continue
        # prepend clean
        if clean_asr_d and clean_sqa_d:
            pts = [(clean_sqa_d["accuracy"], get_asr(clean_asr_d), 0)] + pts
        utils, asrs, sevs = zip(*pts)
        last_sc = ax.scatter(utils, asrs, c=sevs, cmap="YlOrRd", vmin=0, vmax=5,
                             marker=marker, s=100, label=label,
                             edgecolors=ec, linewidths=1.8, zorder=3)
        ax.plot(utils, asrs, color=color, linewidth=1.2, alpha=0.5, zorder=2)
        for u, a, s in pts:
            lbl = "clean" if s == 0 else str(s)
            ax.annotate(lbl, (u, a), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=color)

    if last_sc:
        cb = fig.colorbar(last_sc, ax=ax, label="Severity (0 = clean)")
        cb.set_ticks([0, 1, 2, 3, 4, 5])

    ax.set_xlabel("SQA Utility — Judged Accuracy (%)", labelpad=8)
    ax.set_ylabel("FigStep ASR (%)")
    ax.set_title("Safety–Utility Trade-off Under Image Corruption\n(Base LLaVA-CoT)")
    ax.set_xlim(88, 97)
    ax.set_ylim(58, 76)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(PLOTS, "asr_vs_utility.png")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    plt.close(fig)


if __name__ == "__main__":
    plot_asr_noise()
    plot_orr_sweep()
    plot_model_comparison()
    plot_asr_vs_utility()
    print("\nAll plots saved to", PLOTS)
