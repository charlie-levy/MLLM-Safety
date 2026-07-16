#!/usr/bin/env python3
"""
plot_holisafe_ssu_asr.py — HoliSafe SI+ST->U Conclusion-ASR figure, matching the
SIUO Table-1 style (plot_siuo_asr_6vlm.py). 5 VLMs x 4 image conditions.

Input : the R/C judge summary CSV (columns model,HR_R,HR_C) where `model` is
        "<condition>_<modelkey>" (e.g. clean_llava_cot, zoom_blur_r1_onevision).
        HR_C is the Conclusion ASR (%). Default path is the pulled-local CSV.
Output: figures/holisafe_ssu_conclusion_asr.png  + prints the 5x4 table, the
        per-model delta table, and the summary (degrade-everywhere? worst
        corruption on average? min/max delta).

  python scripts/plot_holisafe_ssu_asr.py
  python scripts/plot_holisafe_ssu_asr.py --csv results/holisafe_ssu/holisafe_ssu_conclusion_asr.csv
"""
import os
import csv
import json
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("base_llama",   "Llama-3.2-V\n(base)"),
    ("llava_cot",    "LLaVA-CoT"),
    ("llamav_o1",    "LlamaV-o1"),
    ("qwen2_5_vl",   "Qwen2.5-VL"),
    ("r1_onevision", "R1-Onevision"),
]
# order + colors identical to the SIUO figure
CONDS = [("clean", "clean", "#8c8c8c"), ("glass_blur", "glass blur", "#4c78a8"),
         ("snow", "snow", "#59a14f"), ("zoom_blur", "zoom blur", "#e15759")]


def load_scores(csv_path):
    """CSV rows model=<cond>_<modelkey>,HR_R,HR_C -> {(cond,modelkey): HR_C}."""
    hrc = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            hrc[row["model"]] = float(row["HR_C"])
    scores = {}
    missing = []
    for ckey, _, _ in CONDS:
        for mkey, _ in MODELS:
            k = "%s_%s" % (ckey, mkey)
            if k in hrc:
                scores[(ckey, mkey)] = hrc[k]
            else:
                missing.append(k)
    if missing:
        print("[WARN] missing cells (not in CSV yet): %s" % missing)
    return scores


def make_figure(scores, out_png):
    fig, ax = plt.subplots(figsize=(15, 7))
    n_m, n_c = len(MODELS), len(CONDS)
    group_w = 0.8
    bar_w = group_w / n_c
    x = np.arange(n_m)
    for j, (ckey, clabel, color) in enumerate(CONDS):
        vals = [scores.get((ckey, mkey), np.nan) for mkey, _ in MODELS]
        offs = x - group_w / 2 + bar_w * (j + 0.5)
        ax.bar(offs, vals, bar_w, color=color, label=clabel, zorder=3,
               edgecolor="white", linewidth=0.5)
        for xi, v in zip(offs, vals):
            if not np.isnan(v):
                ax.text(xi, v + 1.0, "%.1f" % v, ha="center", va="bottom",
                        rotation=90, fontsize=11, color="#222")
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in MODELS], fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Conclusion ASR (%)  —  higher = less safe", fontsize=14)
    ax.set_title("HoliSafe SI+ST→U Conclusion Attack Success Rate under image corruptions — 5 VLMs",
                 fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="condition", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              frameon=False, fontsize=12, title_fontsize=12)
    fig.subplots_adjust(bottom=0.16, top=0.90, left=0.06, right=0.98)
    fig.savefig(out_png, dpi=200)
    print("wrote", out_png)


def print_tables(scores):
    conds = [c[0] for c in CONDS]
    clabels = {c[0]: c[1] for c in CONDS}
    w = 16
    # 5x4 ASR table
    print("\n=== HoliSafe SSU Conclusion ASR (%) — higher = less safe ===")
    print("model".ljust(w) + "".join(clabels[c].rjust(12) for c in conds))
    for mkey, mlbl in MODELS:
        row = "".join(("%.1f" % scores[(c, mkey)]).rjust(12) if (c, mkey) in scores
                      else "   n/a".rjust(12) for c in conds)
        print(mkey.ljust(w) + row)
    # per-model delta table (corruption - clean)
    print("\n=== Delta vs clean (percentage points; + = corruption LESS safe) ===")
    corrs = [c for c in conds if c != "clean"]
    print("model".ljust(w) + "".join(clabels[c].rjust(12) for c in corrs))
    deltas = {c: [] for c in corrs}
    for mkey, _ in MODELS:
        if ("clean", mkey) not in scores:
            print(mkey.ljust(w) + "   (no clean)"); continue
        base = scores[("clean", mkey)]
        cells = ""
        for c in corrs:
            if (c, mkey) in scores:
                d = scores[(c, mkey)] - base
                deltas[c].append(d)
                cells += ("%+.1f" % d).rjust(12)
            else:
                cells += "   n/a".rjust(12)
        print(mkey.ljust(w) + cells)
    # summary
    print("\n=== Summary ===")
    all_d = [d for c in corrs for d in deltas[c]]
    if all_d:
        every = all(d > 0 for d in all_d)
        print("Every model degrades under every corruption: %s" % ("YES" if every else "NO"))
        means = {c: (np.mean(deltas[c]) if deltas[c] else float("nan")) for c in corrs}
        worst = max(means, key=lambda c: means[c])
        print("Mean delta by corruption: " + ", ".join("%s %+.1f" % (clabels[c], means[c]) for c in corrs))
        print("Worst corruption on average: %s (%+.1f pp)" % (clabels[worst], means[worst]))
        print("Delta range across all cells: min %+.1f, max %+.1f" % (min(all_d), max(all_d)))
    else:
        print("(no complete model rows yet)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "results", "holisafe_ssu",
                                                   "holisafe_ssu_conclusion_asr.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "holisafe_ssu_conclusion_asr.png"))
    args = ap.parse_args()
    scores = load_scores(args.csv)
    make_figure(scores, args.out)
    print_tables(scores)


if __name__ == "__main__":
    main()
