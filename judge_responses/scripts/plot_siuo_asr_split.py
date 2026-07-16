#!/usr/bin/env python3
"""Slide 5, split in two: SIUO Conclusion-ASR for REASONING models vs BASE models.
Same metric (HR_conclusion), same conditions, same colors/axis as the 6-VLM figure,
read from figures/siuo_asr_scores.json so numbers never go stale."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(HERE, "figures", "siuo_asr_scores.json")
d = json.load(open(SC))

REASONING = [
    ("llava_cot",            "LLaVA-CoT"),
    ("llamav_o1",            "LlamaV-o1"),
    ("r1_onevision",         "R1-Onevision"),
    ("r1_onevision_nothink", "R1-Onevision\n(no-think)"),
]
BASE = [
    ("base_llama",  "Llama-3.2-V\n(base)"),
    ("qwen2_5_vl",  "Qwen2.5-VL"),
]
CONDS = [("clean", "clean", "#8c8c8c"), ("glass_blur", "glass blur", "#4c78a8"),
         ("snow", "snow", "#59a14f"), ("zoom_blur", "zoom blur", "#e15759")]
METRIC = "HR_conclusion"


def plot(models, title, out, width):
    fig, ax = plt.subplots(figsize=(width, 7))
    n_m, n_c = len(models), len(CONDS)
    group_w = 0.8
    bar_w = group_w / n_c
    x = np.arange(n_m)
    for j, (ckey, clabel, color) in enumerate(CONDS):
        vals = [d[f"siuo_{ckey}_{mkey}"][METRIC] for mkey, _ in models]
        offs = x - group_w / 2 + bar_w * (j + 0.5)
        ax.bar(offs, vals, bar_w, color=color, label=clabel, zorder=3,
               edgecolor="white", linewidth=0.5)
        for xi, v in zip(offs, vals):
            ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", rotation=90,
                    fontsize=11, color="#222")
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in models], fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Conclusion ASR (%)  —  higher = less safe", fontsize=14)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=12)
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="condition", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              frameon=False, fontsize=12, title_fontsize=12)
    fig.subplots_adjust(bottom=0.16, top=0.90, left=0.10, right=0.97)
    fig.savefig(out, dpi=200); print("wrote", out)


plot(REASONING, "Reasoning VLMs — SIUO Conclusion ASR",
     os.path.join(HERE, "figures", "siuo_asr_reasoning.png"), width=11)
plot(BASE, "Base VLMs — SIUO Conclusion ASR",
     os.path.join(HERE, "figures", "siuo_asr_base.png"), width=8)
