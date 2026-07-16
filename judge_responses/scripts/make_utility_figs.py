#!/usr/bin/env python3
"""Two simple slideshow figures: our defenses preserve task utility.
  (1) ScienceQA accuracy under the prompt defense (base LLaVA-CoT, LLaMA-3 judge, n=250).
  (2) MathVista accuracy under the TIS module ablation (official scoring, testmini n=1000).
Numbers are the verified job outputs; each figure carries one takeaway."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "/Users/charlielevy/Desktop/REU/judge_responses/figures"
INK="#1A1A1A"; MUTE="#6b6b6b"; GRAY="#9AA0A6"; BLUE="#3E7CB1"; GRID="#e9e9e9"

def bars(labels, vals, colors, counts, title, sub, baseline, base_note, foot, out, ymax=100):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.62, color=colors, zorder=3, edgecolor="white", linewidth=1)
    ax.axhline(baseline, ls="--", lw=1.3, color="#B23A48", zorder=2)
    # baseline value is already labeled on the reference bar; a text label here collides
    # with that bar's value label, so the dashed line stands on its own.
    for xi, v, c in zip(x, vals, counts):
        ax.text(xi, v+ymax*0.012, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=INK)
        ax.text(xi, v/2, c, ha="center", va="center", fontsize=10.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13, color=INK)
    ax.set_ylim(0, ymax); ax.set_ylabel("Accuracy (%)", fontsize=13, color=INK)
    ax.set_title(title, fontsize=17, fontweight="bold", color=INK, pad=26)
    ax.text(0.5, 1.045, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, color=MUTE)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(colors=INK)
    fig.text(0.5, 0.005, foot, ha="center", fontsize=9.5, color=MUTE)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.11, right=0.97)
    fig.savefig(out, dpi=200, facecolor="white"); print("saved", out)

# (1) ScienceQA — prompt defense
bars(
    labels=["no prompt", "safety\nprompt", "blur-safe\nprompt"],
    vals=[90.8, 89.6, 94.0],
    colors=[GRAY, BLUE, BLUE],
    counts=["227/250", "224/250", "235/250"],
    title="Prompt defense costs no utility",
    sub="ScienceQA accuracy · base LLaVA-CoT",
    baseline=90.8, base_note="– – no-prompt baseline (90.8%)",
    foot="ScienceQA-250 · base LLaVA-CoT · LLaMA-3-8B judge · higher = better",
    out=os.path.join(FIG, "utility_scienceqa_prompting.png"))

# (2) MathVista — TIS module ablation
bars(
    labels=["TIS: both", "TIS: LLM", "TIS: vision"],
    vals=[51.5, 51.1, 50.7],
    colors=[BLUE, BLUE, BLUE],
    counts=["515/1000", "511/1000", "507/1000"],
    title="Where we put TIS barely changes math utility",
    sub="MathVista accuracy · TIS module ablation (all three are TIS variants)",
    baseline=51.1, base_note="– – mean (51.1%)",
    foot="MathVista testmini (1000) · official scoring, Llama-3 extractor · higher = better",
    out=os.path.join(FIG, "utility_mathvista_module.png"))

# (3) R1-Onevision — reasoning on vs off
bars(
    labels=["thinking ON", "thinking OFF"],
    vals=[86.0, 84.0],
    colors=[BLUE, GRAY],
    counts=["215/250", "210/250"],
    title="Turning off reasoning costs little utility",
    sub="R1-Onevision · ScienceQA accuracy · thinking on vs off",
    baseline=86.0, base_note="– – thinking-on (86.0%)",
    foot="ScienceQA-250 · R1-Onevision · paper-exact scoring · higher = better",
    out=os.path.join(FIG, "utility_r1_thinking_scienceqa.png"))

# (4) MathVista — prompt defense (companion to the ScienceQA prompting figure)
bars(
    labels=["no prompt", "safety\nprompt", "blur-safe\nprompt"],
    vals=[53.5, 53.4, 53.0],
    colors=[GRAY, BLUE, BLUE],
    counts=["535/1000", "534/1000", "530/1000"],
    title="Prompt defense costs no utility (math too)",
    sub="MathVista accuracy · base LLaVA-CoT · prompt defense",
    baseline=53.5, base_note="– – no-prompt (53.5%)",
    foot="MathVista testmini (1000) · base LLaVA-CoT · official scoring · higher = better",
    out=os.path.join(FIG, "utility_mathvista_prompting.png"))
