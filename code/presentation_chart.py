#!/usr/bin/env python3
"""
presentation_chart.py — Presentation figure + tables, corrected data.

One panel PER MODEL; x-axis = corruption level (0-80%); three metric lines
(FigStep ASR, Avg ORR, ScienceQA Utility). BIG fonts, WHITE background, BLACK
text, NO blue. Also prints ASR / ORR / Utility tables (rows=model, cols=severity).

Run from repo root:  python3 code/presentation_chart.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "results_newton"; PLOTS = f"{B}/plots"
LEVELS = [0, 20, 40, 60, 80]
MODELS = [("base_tis", "TIS"), ("base_sage", "SAGE"), ("base_msr", "MSR-Align")]
# Non-blue, high-contrast palette on white.
C_ASR, C_ORR, C_SQA = "#C0392B", "#E67E22", "#1E8449"   # red, orange, green

# Big, simple style.
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "text.color": "black", "xtick.color": "black", "ytick.color": "black",
    "font.size": 16, "axes.linewidth": 1.4,
})


def L(p): return json.load(open(p)) if os.path.exists(p) else None


def asr(t, p):
    d = (L(f"{B}/figstep_noise_pct/asr_{t}_gaussian_noise_pct_p{p}.json") if p else
         (L(f"{B}/figstep_noise_pct/asr_{t}_gaussian_noise_pct_p0.json")
          or L(f"{B}/figstep_noise_sweep/asr_{t}_clean.json")))
    return d["asr_pct"] if d else np.nan


def orr(t, kind, p):
    fol = "noise_pct" if kind == "noise" else "blur_pct"
    lab = "gaussian_noise_pct" if kind == "noise" else "gaussian_blur_pct"
    d = (L(f"{B}/orr_{fol}/orr_{t}_{lab}_p{p}.json") if p else
         (L(f"{B}/orr_noise_pct/orr_{t}_gaussian_noise_pct_p0.json") or L(f"{B}/orr/orr_{t}.json")))
    if not d: return (np.nan, np.nan, np.nan)
    return d["xstest"]["orr_pct"], d["mmsa_combined"]["orr_pct"], d["avg_orr_pct"]


def asr_k(t, kind, p):
    if not p: return asr(t, 0)
    fol = "noise_pct" if kind == "noise" else "blur_pct"
    lab = "gaussian_noise_pct" if kind == "noise" else "gaussian_blur_pct"
    d = L(f"{B}/figstep_{fol}/asr_{t}_{lab}_p{p}.json")
    return d["asr_pct"] if d else np.nan


def sqa(t, kind, p):
    if not p:
        d = L(f"{B}/sqa_noise_sweep/judged_{t}_clean.json"); return d["accuracy"] if d else np.nan
    fol = "noise_pct" if kind == "noise" else "blur_pct"
    lab = "gaussian_noise_pct" if kind == "noise" else "gaussian_blur_pct"
    d = L(f"{B}/sqa_{fol}/judged_{t}_{lab}_p{p}.json")
    return d["accuracy"] if d else np.nan


def make_figure(kind):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharey=True)
    for ax, (t, name) in zip(axes, MODELS):
        A = [asr_k(t, kind, p) for p in LEVELS]
        O = [orr(t, kind, p)[2] for p in LEVELS]
        Q = [sqa(t, kind, p) for p in LEVELS]
        ax.plot(LEVELS, A, marker="o", ms=13, lw=4, color=C_ASR, label="FigStep ASR")
        ax.plot(LEVELS, O, marker="^", ms=13, lw=4, color=C_ORR, label="Avg ORR")
        ax.plot(LEVELS, Q, marker="s", ms=13, lw=4, color=C_SQA, label="SQA Utility")
        for series, col in [(A, C_ASR), (O, C_ORR), (Q, C_SQA)]:
            for x, v in zip(LEVELS, series):
                if not np.isnan(v):
                    ax.text(x, v + 3, f"{v:.0f}", ha="center", fontsize=15, fontweight="bold", color=col)
        ax.set_title(name, fontsize=26, fontweight="bold")
        ax.set_xlabel(f"{kind.capitalize()} level (%)", fontsize=20)
        ax.set_xticks(LEVELS); ax.tick_params(labelsize=17)
        ax.set_ylim(-4, 108)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35); ax.set_axisbelow(True)
    axes[0].set_ylabel("Rate / Accuracy (%)", fontsize=20)
    axes[0].legend(fontsize=18, loc="center left", frameon=True)
    fig.suptitle(f"Safety & utility under image {kind} (corrected)", fontsize=27, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{PLOTS}/presentation_{kind}_by_model.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("Saved:", out)


def print_tables():
    hdr = "  " + "".join(f"{p}%".rjust(9) for p in LEVELS)
    for kind in ["noise", "blur"]:
        print("\n" + "=" * 64); print(f"{kind.upper()}"); print("=" * 64)
        print("FigStep ASR (%)  — lower = safer"); print("model     " + hdr)
        for t, n in MODELS:
            print(f"{n:<10}" + "".join(f"{asr_k(t,kind,p):9.1f}" for p in LEVELS))
        print("\nAvg ORR (%)  — lower = less over-refusal  [XSTest+MMSA]/2"); print("model     " + hdr)
        for t, n in MODELS:
            print(f"{n:<10}" + "".join(f"{orr(t,kind,p)[2]:9.1f}" for p in LEVELS))
        print("\nSQA Utility (%)  — higher = better"); print("model     " + hdr)
        for t, n in MODELS:
            print(f"{n:<10}" + "".join(f"{sqa(t,kind,p):9.1f}" for p in LEVELS))


if __name__ == "__main__":
    make_figure("noise")
    make_figure("blur")
    print_tables()
