#!/usr/bin/env python3
"""HoliSafe UI+ST grid: base LLaVA-CoT vs R1-Onevision vs TIS, across clean + 3 corruptions.
UI+ST ASR = string-match compliance rate (paper's SM metric), computed from response files.
Story (mirrors SIUO): both base models sit near-ceiling (~95-98%); only TIS meaningfully
refuses (~60%); image corruption erodes TIS's safety gain (+~4)."""
import sys, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/Users/charlielevy/Desktop/REU/llava_cot_eval/code")
from metrics import is_refusal

RES = "/Users/charlielevy/Desktop/REU/judge_responses/results"
CONDS = [("clean", "clean", "#8c8c8c"), ("zoom_blur", "zoom blur", "#e15759"),
         ("snow", "snow", "#59a14f"), ("glass_blur", "glass blur", "#4c78a8")]
MODELS = [("llava_cot", "Base\nLLaVA-CoT"), ("r1_onevision", "R1-Onevision"), ("tis", "TIS\n(LLaVA-CoT)")]

def path(model, cond):
    if model == "tis":
        return f"{RES}/exp_results/holisafe_tis_clean.jsonl" if cond == "clean" \
               else f"{RES}/part3_results/holisafe_{cond}_tis_responses.jsonl"
    return f"{RES}/part3_results/holisafe_{cond}_{model}_responses.jsonl"

def asr(model, cond):
    rs = [json.loads(l) for l in open(path(model, cond)) if l.strip()]
    resp = [r.get("response") or r.get("full_response") or "" for r in rs]
    hint = "tis" if model == "tis" else None
    return 100.0 * sum(0 if is_refusal(x, hint) else 1 for x in resp) / len(resp)

grid = {(m, c): asr(m, c) for m, _ in MODELS for c, _, _ in CONDS}
N = 494
INK = "#1A1A1A"; MUTE = "#6b6b6b"

fig, ax = plt.subplots(figsize=(10.5, 6.2))
n_m, n_c = len(MODELS), len(CONDS)
gw, bw = 0.82, 0.82 / n_c
x = np.arange(n_m)
for j, (ckey, clabel, color) in enumerate(CONDS):
    vals = [grid[(mkey, ckey)] for mkey, _ in MODELS]
    offs = x - gw/2 + bw*(j+0.5)
    ax.bar(offs, vals, bw, color=color, label=clabel, zorder=3, edgecolor="white", linewidth=0.6)
    for xi, v in zip(offs, vals):
        ax.text(xi, v+1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color="#333", rotation=0)

ax.axhspan(90, 100, color="#f3d9d9", alpha=0.5, zorder=0)   # ceiling band
ax.text(2.42, 92.5, "near-ceiling\n(barely refuses)", ha="right", va="center",
        fontsize=9.5, color="#a23", style="italic", linespacing=1.05)

ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in MODELS], fontsize=13, color=INK)
ax.set_ylim(0, 104); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("UI+ST Attack Success Rate (%)  —  lower = safer", fontsize=13, color=INK)
ax.set_title("HoliSafe UI+ST: only TIS refuses — and corruption erodes it",
             fontsize=15.5, fontweight="bold", color=INK, pad=12)
ax.legend(title="condition", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08),
          frameon=False, fontsize=11, title_fontsize=11)
ax.grid(axis="y", color="#ececec", lw=1, zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(colors=INK)
fig.text(0.5, 0.005, f"HoliSafe UI+ST · n={N} · string-match ASR (SM metric) · base models near-ceiling; "
         f"TIS cuts ASR ~35 pts; corruption pushes TIS back up ~4",
         ha="center", fontsize=9.5, color=MUTE)
fig.subplots_adjust(bottom=0.17, top=0.91, left=0.08, right=0.97)
out = f"{RES}/../figures/holisafe_grid_asr.png"
fig.savefig(out, dpi=200, facecolor="white"); print("saved", os.path.abspath(out))
for m, ml in MODELS:
    print(f"  {m:14s}: " + "  ".join(f"{c}={grid[(m,c)]:.1f}" for c,_,_ in CONDS))
