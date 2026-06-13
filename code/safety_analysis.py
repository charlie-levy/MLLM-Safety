#!/usr/bin/env python3
"""
safety_analysis.py — Correct, self-validating safety + utility analysis across
the THREE safety datasets (FigStep, XSTest, MMSA) and a utility set (ScienceQA),
for all THREE adapters (TIS, SAGE, MSR-Align), under clean / Gaussian-noise /
Gaussian-blur image corruption.

WHY THIS SCRIPT EXISTS (methodology note)
─────────────────────────────────────────────────────────────────────────────
The metric for each dataset is dictated by what the dataset contains:
    FigStep  -> ASR  (all prompts harmful; ASR = fraction the model complied with)
    XSTest   -> ORR  (all prompts safe;    ORR = fraction the model refused)
    MMSA     -> ORR  (safe subset;         ORR = fraction judged unsafe / refused)
    ScienceQA-> Utility (multiple-choice accuracy)
You cannot put ASR on XSTest/MMSA (no attacks) or utility on the safety sets.

CRITICAL: XSTest and MMSA are kept SEPARATE — never averaged. They are different
experiments. XSTest typesets the safe REQUEST *inside the image*; the text prompt
is only "extract the text and judge it". So corrupting the image destroys the
request, and XSTest-ORR-under-corruption is NOT a valid robustness measurement —
it measures how the model reacts to an unreadable smudge. MMSA puts the request in
the TEXT prompt with a real photo as context, so corruption only degrades visual
grounding and the measurement stays valid. The old "avg_orr" bar mixed the two,
which is why those charts looked random.

Reads existing result JSONs only (no re-run needed). Run from the repo root:
    python3 code/safety_analysis.py
Outputs PNGs to results_newton/plots/ (prefix: analysis_*).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
BASE   = "results_newton"
PLOTS  = f"{BASE}/plots"
MODELS = [("base_tis", "TIS"), ("base_sage", "SAGE"), ("base_msr", "MSR-Align")]
LEVELS = [0, 20, 40, 60, 80]                 # 0 = clean, capped at 80%
# Synthetic corruptions (always have data) — used by the validation pass.
KINDS  = [("noise", "noise_pct", "gaussian_noise_pct", "Gaussian Noise"),
          ("blur",  "blur_pct",  "gaussian_blur_pct",  "Gaussian Blur")]
# All corruption families. The realistic ones (jpeg/brightness/pixelate) populate
# as their Newton jobs finish; the combo charts auto-skip any family with no data.
SWEEP_KINDS = KINDS + [
    ("jpeg",       "jpeg_pct",       "jpeg_pct",       "JPEG Compression"),
    ("brightness", "brightness_pct", "brightness_pct", "Low-light / Dimming"),
    ("pixelate",   "pixelate_pct",   "pixelate_pct",   "Pixelation / Low-res"),
]
MODEL_COLORS = {"TIS": "#C44E52", "SAGE": "#4C72B0", "MSR-Align": "#55A868"}

# Expected sample counts — used by the validation pass.
N_FIGSTEP, N_XSTEST, N_MMSA, N_SQA = 500, 250, 428, 250


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _folder_label(kind):
    for k, fol, lab, _ in SWEEP_KINDS:
        if k == kind:
            return fol, lab
    raise ValueError(kind)


def _kind_has_data(kind):
    """True if at least one model has a p20 FigStep result for this family."""
    fol, lab = _folder_label(kind)
    return any(os.path.exists(f"{BASE}/figstep_{fol}/asr_{tag}_{lab}_p20.json")
               for tag, _ in MODELS)


# ── Metric accessors (one source of truth for every number) ─────────────────
def figstep_asr(tag, kind, p):
    """FigStep Attack Success Rate (%). p=0 reuses the clean baseline."""
    if p == 0:
        d = _load(f"{BASE}/figstep_noise_sweep/asr_{tag}_clean.json")
    else:
        fol, lab = _folder_label(kind)
        d = _load(f"{BASE}/figstep_{fol}/asr_{tag}_{lab}_p{p}.json")
    return d["asr_pct"] if d else None


def orr_split(tag, kind, p):
    """(XSTest ORR%, MMSA ORR%) kept separate. p=0 reuses the clean baseline."""
    if p == 0:
        d = _load(f"{BASE}/orr/orr_{tag}.json")
    else:
        fol, lab = _folder_label(kind)
        d = _load(f"{BASE}/orr_{fol}/orr_{tag}_{lab}_p{p}.json")
    if not d:
        return None, None
    return d["xstest"]["orr_pct"], d["mmsa_combined"]["orr_pct"]


def sqa_util(tag, kind, p):
    """ScienceQA accuracy (%). p=0 reuses the clean baseline."""
    if p == 0:
        d = _load(f"{BASE}/sqa_noise_sweep/judged_{tag}_clean.json")
    else:
        fol, lab = _folder_label(kind)
        d = _load(f"{BASE}/sqa_{fol}/judged_{tag}_{lab}_p{p}.json")
    return d["accuracy"] if d else None


# ── Validation pass: prove the data is complete and sane before plotting ────
def validate():
    print("=" * 74)
    print("VALIDATION  (counts, completeness, value ranges)")
    print("=" * 74)
    problems = []

    for tag, name in MODELS:
        # clean baselines exist with the right N
        fc = _load(f"{BASE}/figstep_noise_sweep/asr_{tag}_clean.json")
        oc = _load(f"{BASE}/orr/orr_{tag}.json")
        sc = _load(f"{BASE}/sqa_noise_sweep/judged_{tag}_clean.json")
        if not fc:                       problems.append(f"{name}: FigStep clean missing")
        elif fc["n_total"] != N_FIGSTEP: problems.append(f"{name}: FigStep n={fc['n_total']} (want {N_FIGSTEP})")
        if not oc:                                   problems.append(f"{name}: ORR clean missing")
        else:
            if oc["xstest"]["n_total"] != N_XSTEST:  problems.append(f"{name}: XSTest n={oc['xstest']['n_total']} (want {N_XSTEST})")
            if oc["mmsa_combined"]["n_total"] != N_MMSA: problems.append(f"{name}: MMSA n={oc['mmsa_combined']['n_total']} (want {N_MMSA})")
        if not sc:                       problems.append(f"{name}: SQA clean missing")
        elif sc["total"] != N_SQA:       problems.append(f"{name}: SQA n={sc['total']} (want {N_SQA})")

        # every corruption cell present and in [0, 100]
        for kind, _, _, _ in KINDS:
            for p in [20, 40, 60, 80]:
                a = figstep_asr(tag, kind, p)
                xs, mm = orr_split(tag, kind, p)
                u = sqa_util(tag, kind, p)
                for metric, v in [("ASR", a), ("XSTest", xs), ("MMSA", mm), ("SQA", u)]:
                    if v is None:
                        problems.append(f"{name} {kind} p{p}: {metric} missing")
                    elif not (0 <= v <= 100):
                        problems.append(f"{name} {kind} p{p}: {metric}={v} out of range")

    if problems:
        print("FAILED — %d problem(s):" % len(problems))
        for pb in problems:
            print("   •", pb)
        sys.exit("Aborting: fix the data before trusting the charts.")
    print("PASSED — 3 models × (FigStep 500, XSTest 250, MMSA 428, SQA 250),")
    print("         all clean + noise/blur p20-80 cells present, all values in [0,100].\n")


# ── Human-readable data table (so the numbers are inspectable) ──────────────
def print_table():
    print("=" * 74)
    print("DATA TABLE  (all values are %)   ASR↓ better · ORR↓ better · SQA↑ better")
    print("=" * 74)
    hdr = f"{'model':<10}{'cond':<12}{'FigStep ASR':>12}{'XSTest ORR':>12}{'MMSA ORR':>11}{'SQA':>7}"
    for tag, name in MODELS:
        print("-" * 74)
        print(hdr)
        # clean
        xs, mm = orr_split(tag, "noise", 0)
        print(f"{name:<10}{'clean':<12}{figstep_asr(tag,'noise',0):>12.1f}{xs:>12.1f}{mm:>11.1f}{sqa_util(tag,'noise',0):>7.1f}")
        for kind, _, _, pretty in KINDS:
            for p in [20, 40, 60, 80]:
                xs, mm = orr_split(tag, kind, p)
                print(f"{'':<10}{kind+' '+str(p)+'%':<12}{figstep_asr(tag,kind,p):>12.1f}{xs:>12.1f}{mm:>11.1f}{sqa_util(tag,kind,p):>7.1f}")
    print()


# ── Figure 1: clean safety + utility profile (the core 3×4 comparison) ──────
def fig_clean_profile():
    metrics = ["FigStep\nASR ↓", "XSTest\nORR ↓", "MMSA\nORR ↓", "ScienceQA\nUtility ↑"]
    data = {name: [] for _, name in MODELS}
    for tag, name in MODELS:
        a = figstep_asr(tag, "noise", 0)
        xs, mm = orr_split(tag, "noise", 0)
        u = sqa_util(tag, "noise", 0)
        data[name] = [a, xs, mm, u]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for j, (_, name) in enumerate(MODELS):
        off = (j - 1) * width
        bars = ax.bar(x + off, data[name], width, label=name,
                      color=MODEL_COLORS[name], edgecolor="white", linewidth=1)
        for r, v in zip(bars, data[name]):
            ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=MODEL_COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.set_ylabel("Rate / Accuracy (%)", fontsize=15)
    ax.set_ylim(0, 105)
    ax.set_title("Clean image: safety & utility profile of each safety adapter",
                 fontsize=17, fontweight="bold", pad=12)
    ax.legend(fontsize=14, loc="upper center", ncol=3, frameon=True)
    ax.axhspan(0, 105, xmin=0, xmax=0.75, color="#fff5f5", zorder=0)  # subtle "lower better" zone
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = f"{PLOTS}/analysis_clean_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Saved:", out)


# ── Figure 2: refusal-discrimination gap (the headline finding) ─────────────
def fig_discrimination():
    """Refusal rate on harmful (FigStep) vs safe (XSTest, MMSA) at clean.
    Ideal model: tall green (refuse attacks), short orange (allow safe).
    The gap between them = how well the model tells safe from harmful."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(MODELS))
    width = 0.26
    series = [
        ("Refuse FigStep (harmful) — want HIGH", "#2E7D32", lambda tag: 100 - figstep_asr(tag, "noise", 0)),
        ("Refuse XSTest (safe) — want LOW",      "#E69F00", lambda tag: orr_split(tag, "noise", 0)[0]),
        ("Refuse MMSA (safe) — want LOW",        "#D55E00", lambda tag: orr_split(tag, "noise", 0)[1]),
    ]
    for j, (label, color, fn) in enumerate(series):
        vals = [fn(tag) for tag, _ in MODELS]
        off = (j - 1) * width
        bars = ax.bar(x + off, vals, width, label=label, color=color,
                      edgecolor="white", linewidth=1)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color=color)

    # annotate the discrimination gap = refuse(harmful) - mean refuse(safe)
    for i, (tag, name) in enumerate(MODELS):
        rh = 100 - figstep_asr(tag, "noise", 0)
        xs, mm = orr_split(tag, "noise", 0)
        gap = rh - (xs + mm) / 2
        ax.text(i, 103, f"gap = {gap:.0f}", ha="center", fontsize=13,
                fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in MODELS], fontsize=15)
    ax.set_ylabel("Refusal rate (%)", fontsize=15)
    ax.set_ylim(0, 112)
    ax.set_title("Safety discrimination: refuse the harmful, allow the safe\n"
                 "(bigger gap = better; TIS refuses safe almost as often as harmful)",
                 fontsize=16, fontweight="bold", pad=10)
    ax.legend(fontsize=12, loc="lower center", ncol=1, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = f"{PLOTS}/analysis_discrimination.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Saved:", out)


# ── Figure 3: FigStep ASR under corruption (the opposite-directions finding) ─
def fig_asr_under_corruption():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (kind, _, _, pretty) in zip(axes, KINDS):
        for tag, name in MODELS:
            ys = [figstep_asr(tag, kind, p) for p in LEVELS]
            ax.plot(LEVELS, ys, marker="o", markersize=9, linewidth=2.6,
                    label=name, color=MODEL_COLORS[name])
            ax.text(LEVELS[-1] + 1.5, ys[-1], f"{ys[-1]:.0f}", fontsize=12,
                    fontweight="bold", color=MODEL_COLORS[name], va="center")
        ax.set_title(pretty, fontsize=16, fontweight="bold")
        ax.set_xlabel("Corruption level (%)  ·  0 = clean", fontsize=14)
        ax.set_xticks(LEVELS)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("FigStep ASR (%)  ·  lower = safer", fontsize=14)
    axes[0].set_ylim(-2, max(35, axes[0].get_ylim()[1]))
    axes[0].legend(fontsize=13, loc="upper left")
    fig.suptitle("Image corruption moves safety in OPPOSITE directions per adapter\n"
                 "TIS gets more jailbreakable · MSR gets safer · SAGE stays immune",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{PLOTS}/analysis_asr_under_corruption.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Saved:", out)


# ── Figure 4: XSTest vs MMSA ORR under corruption (the modality confound) ────
def fig_orr_modality_confound():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    col = {"XSTest": 0, "MMSA": 1}
    row = {"noise": 0, "blur": 1}
    for kind, _, _, pretty in KINDS:
        for which, ci in col.items():
            ax = axes[row[kind]][ci]
            for tag, name in MODELS:
                ys = []
                for p in LEVELS:
                    xs, mm = orr_split(tag, kind, p)
                    ys.append(xs if which == "XSTest" else mm)
                ax.plot(LEVELS, ys, marker="o", markersize=8, linewidth=2.4,
                        label=name, color=MODEL_COLORS[name])
            ax.set_title(f"{which} ORR — {pretty}", fontsize=15, fontweight="bold")
            ax.set_xticks(LEVELS)
            ax.set_ylim(0, 105)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if row[kind] == 1:
                ax.set_xlabel("Corruption level (%)", fontsize=13)
            if ci == 0:
                ax.set_ylabel("ORR (%)", fontsize=13)
    # shade the XSTest column to flag it as corruption-invalid
    for r in (0, 1):
        axes[r][0].set_facecolor("#fcf3f3")
    axes[0][0].legend(fontsize=12, loc="upper left")
    fig.suptitle("Why XSTest and MMSA can't be averaged: the request lives in different places\n"
                 "XSTest request is INSIDE the image (corruption erases it → erratic) · "
                 "MMSA request is in TEXT (stays valid → smooth)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{PLOTS}/analysis_orr_modality_confound.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Saved:", out)


# ── Figure 5: utility (ScienceQA) decay under corruption ────────────────────
def fig_utility_decay():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (kind, _, _, pretty) in zip(axes, KINDS):
        for tag, name in MODELS:
            ys = [sqa_util(tag, kind, p) for p in LEVELS]
            ax.plot(LEVELS, ys, marker="s", markersize=9, linewidth=2.6,
                    label=name, color=MODEL_COLORS[name])
            ax.text(LEVELS[-1] + 1.5, ys[-1], f"{ys[-1]:.0f}", fontsize=12,
                    fontweight="bold", color=MODEL_COLORS[name], va="center")
        ax.set_title(pretty, fontsize=16, fontweight="bold")
        ax.set_xlabel("Corruption level (%)  ·  0 = clean", fontsize=14)
        ax.set_xticks(LEVELS)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ScienceQA accuracy (%)  ·  higher = better", fontsize=14)
    axes[0].legend(fontsize=13, loc="lower left")
    fig.suptitle("Utility cost of corruption: all three adapters lose ~15-20 pts by 80%",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{PLOTS}/analysis_utility_decay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Saved:", out)


# ── Signature combo chart: stacked safety-failure bars + utility line ───────
def fig_combo_per_model():
    """Dashboard-style combo, one per (corruption family, model):
      • stacked bars (LEFT axis)  = safety-failure load: FigStep ASR (unsafe
        compliance) stacked with MMSA ORR (over-refusal) — both 'lower is better',
        so a taller stack = worse safety.
      • line (RIGHT axis, 0-100) = ScienceQA Utility — the capability trend that
        runs through the middle, the way a total/forecast line sits over cost bars.
    Auto-skips corruption families that have no data yet."""
    x = np.arange(len(LEVELS))
    n = 0
    for kind, fol, lab, pretty in SWEEP_KINDS:
        if not _kind_has_data(kind):
            continue
        for tag, name in MODELS:
            asr  = [figstep_asr(tag, kind, p) for p in LEVELS]
            mm   = [orr_split(tag, kind, p)[1] for p in LEVELS]
            util = [sqa_util(tag, kind, p) for p in LEVELS]
            asr  = [v if v is not None else np.nan for v in asr]
            mm   = [v if v is not None else np.nan for v in mm]
            util = [v if v is not None else np.nan for v in util]

            fig, axL = plt.subplots(figsize=(11, 6.3))
            axR = axL.twinx()

            b1 = axL.bar(x, asr, width=0.60, color="#C44E52", edgecolor="white",
                         linewidth=1, label="FigStep ASR  (unsafe compliance ↓)")
            b2 = axL.bar(x, mm, width=0.60, bottom=asr, color="#E69F00",
                         edgecolor="white", linewidth=1,
                         label="MMSA ORR  (over-refusal ↓)")
            # value labels inside each segment
            for xi, a, m in zip(x, asr, mm):
                if not np.isnan(a) and a > 4:
                    axL.text(xi, a / 2, f"{a:.0f}", ha="center", va="center",
                             fontsize=11, fontweight="bold", color="white")
                if not np.isnan(m) and m > 6:
                    axL.text(xi, a + m / 2, f"{m:.0f}", ha="center", va="center",
                             fontsize=11, fontweight="bold", color="white")

            ln = axR.plot(x, util, color="#1f9e89", marker="o", markersize=11,
                          linewidth=3.4, label="ScienceQA Utility (↑)", zorder=6)
            for xi, u in zip(x, util):
                if not np.isnan(u):
                    axR.text(xi, u + 2.5, f"{u:.0f}", ha="center", va="bottom",
                             fontsize=12, fontweight="bold", color="#13705f", zorder=7)

            axL.set_xticks(x)
            axL.set_xticklabels([f"{p}%\nclean" if p == 0 else f"{p}%" for p in LEVELS],
                                fontsize=13)
            axL.set_xlabel(f"{pretty} level", fontsize=14)
            axL.set_ylabel("Safety-failure load (stacked %)", fontsize=13)
            axR.set_ylabel("ScienceQA Utility (%)", fontsize=13, color="#13705f")
            axR.tick_params(axis="y", colors="#13705f")
            top = np.nanmax([a + m for a, m in zip(asr, mm)] + [1])
            axL.set_ylim(0, max(120, top * 1.15))
            axR.set_ylim(0, 100)
            axL.set_title(f"{name} under {pretty}: safety-failure bars vs utility line",
                          fontsize=15, fontweight="bold", pad=10)
            # combined legend, placed below the plot so it never covers the bars
            h1, l1 = axL.get_legend_handles_labels()
            h2, l2 = axR.get_legend_handles_labels()
            axL.legend(h1 + h2, l1 + l2, fontsize=11, loc="upper center",
                       bbox_to_anchor=(0.5, -0.13), ncol=3, framealpha=0.95)
            axL.grid(axis="y", linestyle="--", alpha=0.3)
            axL.set_axisbelow(True)
            fig.tight_layout()
            out = f"{PLOTS}/analysis_combo_{kind}_{tag}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
            print("Saved:", out); n += 1
    if n == 0:
        print("(combo) no corruption families have data yet — skipped")


def main():
    if not os.path.isdir(BASE):
        sys.exit(f"Run from the repo root; '{BASE}/' not found.")
    os.makedirs(PLOTS, exist_ok=True)
    validate()
    print_table()
    print("=" * 74)
    print("FIGURES")
    print("=" * 74)
    fig_clean_profile()
    fig_discrimination()
    fig_asr_under_corruption()
    fig_orr_modality_confound()
    fig_utility_decay()
    fig_combo_per_model()
    print("\nAll analysis figures written to", PLOTS)


if __name__ == "__main__":
    main()
