#!/usr/bin/env python3
"""
SIUO ASR results (advisor's run) — grouped bar plot.
Metrics: ASR_reasoning, ASR_conclusion (SIUO Attack Success Rate, %; higher = LESS safe).
Raw JSON keys are "HR_*" but HR == ASR here (per advisor).
Data transcribed verbatim from the advisor's "Raw scores" JSON.
4 models x 4 conditions x 2 metrics.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- data (transcribed exactly from the advisor's Raw scores) ----
RAW = {
    "siuo_clean_base_llama":        {"HR_reasoning": 61.08, "HR_conclusion": 60.48},
    "siuo_clean_llava_cot":         {"HR_reasoning": 69.46, "HR_conclusion": 68.26},
    "siuo_clean_qwen2_5_vl":        {"HR_reasoning": 70.06, "HR_conclusion": 70.06},
    "siuo_clean_r1_onevision":      {"HR_reasoning": 86.23, "HR_conclusion": 83.83},
    "siuo_glass_blur_base_llama":   {"HR_reasoning": 60.48, "HR_conclusion": 61.08},
    "siuo_glass_blur_llava_cot":    {"HR_reasoning": 77.25, "HR_conclusion": 71.26},
    "siuo_glass_blur_qwen2_5_vl":   {"HR_reasoning": 72.46, "HR_conclusion": 73.05},
    "siuo_glass_blur_r1_onevision": {"HR_reasoning": 88.02, "HR_conclusion": 85.63},
    "siuo_snow_base_llama":         {"HR_reasoning": 66.47, "HR_conclusion": 64.67},
    "siuo_snow_llava_cot":          {"HR_reasoning": 77.84, "HR_conclusion": 71.86},
    "siuo_snow_qwen2_5_vl":         {"HR_reasoning": 73.65, "HR_conclusion": 73.05},
    "siuo_snow_r1_onevision":       {"HR_reasoning": 87.43, "HR_conclusion": 84.43},
    "siuo_zoom_blur_base_llama":    {"HR_reasoning": 68.26, "HR_conclusion": 69.46},
    "siuo_zoom_blur_llava_cot":     {"HR_reasoning": 79.04, "HR_conclusion": 74.85},
    "siuo_zoom_blur_qwen2_5_vl":    {"HR_reasoning": 74.85, "HR_conclusion": 74.25},
    "siuo_zoom_blur_r1_onevision":  {"HR_reasoning": 89.82, "HR_conclusion": 86.83},
}

CONDITIONS = ["clean", "glass_blur", "snow", "zoom_blur"]
MODELS = ["base_llama", "llava_cot", "qwen2_5_vl", "r1_onevision"]
MODEL_LABEL = {
    "base_llama": "Llama-3.2-V\n(base)",
    "llava_cot": "LLaVA-CoT",
    "qwen2_5_vl": "Qwen2.5-VL",
    "r1_onevision": "R1-Onevision",
}
COND_LABEL = {"clean": "clean", "glass_blur": "glass blur", "snow": "snow", "zoom_blur": "zoom blur"}
COND_COLOR = {"clean": "#9aa0a6", "glass_blur": "#4c78a8", "snow": "#54a24b", "zoom_blur": "#e45756"}

def val(cond, model, metric):
    return RAW[f"siuo_{cond}_{model}"][metric]

# ---- single panel: Conclusion ASR only (reasoning ignored per request) ----
METRIC = "HR_conclusion"          # raw JSON key; HR == ASR (Attack Success Rate)

fig, ax = plt.subplots(figsize=(10, 6.2))
x = np.arange(len(MODELS))
nbar = len(CONDITIONS)
width = 0.8 / nbar

for i, cond in enumerate(CONDITIONS):
    offs = (i - (nbar - 1) / 2) * width
    vals = [val(cond, m, METRIC) for m in MODELS]
    bars = ax.bar(x + offs, vals, width, color=COND_COLOR[cond],
                  edgecolor="white", linewidth=0.6, label=COND_LABEL[cond])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8.5, rotation=90, color="#333")

ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=10.5)
ax.set_ylim(0, 100)
ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_ylabel("Conclusion ASR (%)  —  higher = less safe", fontsize=11)

handles = [Patch(facecolor=COND_COLOR[c], label=COND_LABEL[c]) for c in CONDITIONS]
ax.legend(handles=handles, ncol=4, loc="upper center", frameon=False,
          bbox_to_anchor=(0.5, -0.08), fontsize=10.5, title="condition", title_fontsize=10.5)
fig.suptitle("SIUO Conclusion Attack Success Rate under image corruptions — 4 VLMs",
             fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])

out = "figures/siuo_conclusion_asr_corruptions.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)

# also dump the transcribed data as a clean JSON alongside it (HR keys == ASR)
with open("figures/siuo_asr_scores.json", "w") as f:
    json.dump(RAW, f, indent=2)
print("wrote figures/siuo_asr_scores.json")
