#!/usr/bin/env python3
"""
plots.py — all figures for the Part-4 exploratory analysis. Reads the CSVs written
by build_master.py / consistency.py / cluster.py (and recomputes the master frame
for distribution plots). Saves PNGs to outputs/figures/.

Each figure's title states the plain-language takeaway.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lib

lib.ensure_dirs()
F = lib.FIGS
df = pd.DataFrame(lib.load_records())
CON, MOD = lib.CONDITIONS, lib.MODELS
LAB = {m: lib.MODEL_META[m]["label"] for m in MOD}
CL = lib.COLOR
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(F, name), bbox_inches="tight")
    plt.close(fig)
    print("  figures/%s" % name)


def heat(ax, M, title, fmt="%.0f", cmap="viridis"):
    im = ax.imshow(M, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(CON))); ax.set_xticklabels(CON, rotation=30, ha="right")
    ax.set_yticks(range(len(MOD))); ax.set_yticklabels([LAB[m] for m in MOD])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                    color="white" if im.norm(M[i, j]) < 0.55 else "black", fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def matrix(metric, agg="mean"):
    p = df.pivot_table(index="model", columns="condition", values=metric, aggfunc=agg)
    return p.reindex(index=MOD, columns=CON).values


# 1) multi-panel metric heatmaps -------------------------------------------------
panels = [("answer_words", "Mean ANSWER length (words)", "%.0f", "viridis"),
          ("reasoning_words", "Mean REASONING length (words)", "%.0f", "magma"),
          ("perception_failure", "Perception-failure rate", "%.2f", "Reds"),
          ("refusal_like", "Refusal-like phrasing rate\n(surface marker, not safety)", "%.2f", "Purples"),
          ("uncertainty_density", "Uncertainty / 100 answer words", "%.2f", "Blues"),
          ("bigram_repeat", "Bigram-repeat (mode-collapse proxy)", "%.2f", "Oranges")]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (m, t, fmt, cm) in zip(axes.ravel(), panels):
    im = heat(ax, matrix(m), t, fmt, cm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Part 4 — surface-metric heatmaps (model × image corruption, SIUO n=167)", fontsize=11)
save(fig, "01_metric_heatmaps.png")

# 2) answer-length violins per model x condition --------------------------------
fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
for ax, m in zip(axes, MOD):
    data = [df[(df.model == m) & (df.condition == c)]["answer_words"].values for c in CON]
    vp = ax.violinplot(data, showmedians=True, widths=0.85)
    for b in vp["bodies"]:
        b.set_facecolor(CL[m]); b.set_alpha(0.6)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(CON, rotation=30, ha="right")
    ax.set_title(LAB[m], fontsize=9)
    ax.set_ylim(0, 400)
axes[0].set_ylabel("answer words")
fig.suptitle("Answer-length distribution per model across corruptions "
             "(median bar) — width = spread, not just the mean", fontsize=11)
save(fig, "02_answer_length_violins.png")

# 3) length trajectory clean -> corruptions -------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in MOD:
    y = [df[(df.model == m) & (df.condition == c)]["answer_words"].mean() for c in CON]
    ax.plot(CON, y, "o-", color=CL[m], label=LAB[m], lw=2)
ax.set_ylabel("mean answer words"); ax.set_xlabel("image condition")
ax.set_title("Verbosity drift: most models get SLIGHTLY LONGER under corruption\n(no length collapse)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
save(fig, "03_length_trajectory.png")

# 4) reasoning vs answer length (the two reasoning models) ----------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, m in zip(axes, ["llava_cot", "r1_onevision"]):
    rw = [df[(df.model == m) & (df.condition == c)]["reasoning_words"].mean() for c in CON]
    aw = [df[(df.model == m) & (df.condition == c)]["answer_words"].mean() for c in CON]
    x = np.arange(len(CON))
    ax.bar(x - 0.2, rw, 0.4, label="reasoning", color="#888")
    ax.bar(x + 0.2, aw, 0.4, label="final answer", color=CL[m])
    ax.set_xticks(x); ax.set_xticklabels(CON, rotation=30, ha="right")
    ax.set_title(LAB[m], fontsize=9); ax.legend(fontsize=8)
axes[0].set_ylabel("mean words")
fig.suptitle("Reasoning does NOT collapse under corruption — if anything it inflates "
             "(esp. LLaVA-CoT)", fontsize=11)
save(fig, "04_reasoning_vs_answer.png")

# 5) content-drift cosine heatmap (clean vs corrupted) --------------------------
ds = pd.read_csv(os.path.join(lib.TABLES, "drift_summary.csv"))
corr = ["zoom_blur", "snow", "glass_blur"]
Mc = ds.pivot(index="model", columns="condition", values="mean_cosine").reindex(index=MOD, columns=corr).values
fig, ax = plt.subplots(figsize=(6.2, 4))
im = ax.imshow(Mc, aspect="auto", cmap="viridis_r", vmin=0.35, vmax=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(corr, rotation=20, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels([LAB[m] for m in MOD])
for i in range(4):
    for j in range(3):
        ax.text(j, i, "%.2f" % Mc[i, j], ha="center", va="center", color="white", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Content drift: TF-IDF cosine(clean, corrupted)\nLOWER = answer moved MORE "
             "(zoom_blur moves it most; Qwen-base moves least)", fontsize=9)
save(fig, "05_drift_cosine_heatmap.png")

# 6) drift buckets stacked -------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.4))
labels, stab, part, chg = [], [], [], []
for m in MOD:
    for c in corr:
        r = ds[(ds.model == m) & (ds.condition == c)].iloc[0]
        labels.append("%s\n%s" % (m.replace("_", " "), c)); stab.append(r["stable"]); part.append(r["partial"]); chg.append(r["changed"])
x = np.arange(len(labels))
ax.bar(x, stab, color="#2c7", label="stable (cos≥.6)")
ax.bar(x, part, bottom=stab, color="#fc6", label="partial")
ax.bar(x, chg, bottom=np.array(stab) + np.array(part), color="#e55", label="changed (cos<.3)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
ax.set_ylabel("fraction of prompts"); ax.legend(fontsize=8, ncol=3, loc="lower center")
ax.set_title("Per-prompt stability clean→corrupted: Qwen-base most stable, "
             "reasoning models most 'partial'", fontsize=10)
save(fig, "06_drift_buckets.png")

# 7) the two failure modes: perception-failure vs silent content drift ----------
fig, ax = plt.subplots(figsize=(7, 5))
for m in MOD:
    for c in corr:
        pf = df[(df.model == m) & (df.condition == c)]["perception_failure"].mean()
        cos = ds[(ds.model == m) & (ds.condition == c)]["mean_cosine"].iloc[0]
        ax.scatter(1 - cos, pf, s=90, color=CL[m], edgecolor="k", lw=0.5,
                   marker={"zoom_blur": "o", "snow": "s", "glass_blur": "^"}[c])
        ax.annotate(c[:4], (1 - cos, pf), fontsize=6, xytext=(3, 3), textcoords="offset points")
for m in MOD:
    ax.scatter([], [], color=CL[m], label=LAB[m])
for c, mk in [("zoom_blur", "o"), ("snow", "s"), ("glass_blur", "^")]:
    ax.scatter([], [], marker=mk, color="gray", label=c)
ax.set_xlabel("silent content drift  (1 − cosine)")
ax.set_ylabel("overt perception-failure rate")
ax.set_title("Two distinct failure modes of corruption:\nglass_blur → 'I can't see it' (up);  "
             "zoom_blur → confidently DIFFERENT answer (right)", fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
save(fig, "07_two_failure_modes.png")

# 8) LSA scatter colored by model -----------------------------------------------
co = pd.read_csv(os.path.join(lib.OUT, "pca_coords.csv"))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for m in MOD:
    s = co[co.model == m]
    axes[0].scatter(s.pc1, s.pc2, s=6, alpha=0.4, color=CL[m], label=LAB[m])
axes[0].legend(fontsize=7, markerscale=2); axes[0].set_title("answers colored by MODEL", fontsize=9)
cmap = {"clean": "#999", "zoom_blur": "#39f", "snow": "#3c9", "glass_blur": "#e53"}
for c in CON:
    s = co[co.condition == c]
    axes[1].scatter(s.pc1, s.pc2, s=6, alpha=0.4, color=cmap[c], label=c)
axes[1].legend(fontsize=7, markerscale=2); axes[1].set_title("same points colored by CONDITION", fontsize=9)
for ax in axes:
    ax.set_xlabel("LSA-1"); ax.set_ylabel("LSA-2")
fig.suptitle("Answer space separates by MODEL but NOT by corruption "
             "(corruption ≠ a distinct output region)", fontsize=11)
save(fig, "08_lsa_scatter.png")

# 9) radar of model style profiles (clean condition, min-max normalized) --------
metrics = ["answer_words", "reasoning_ratio", "connector_density", "uncertainty_density",
           "refusal_like", "mattr_answer", "bigram_repeat"]
clean = df[df.condition == "clean"].groupby("model")[metrics].mean()
norm = (clean - clean.min()) / (clean.max() - clean.min() + 1e-9)
ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
for m in MOD:
    v = norm.loc[m].tolist(); v += v[:1]
    ax.plot(ang, v, color=CL[m], lw=2, label=LAB[m]); ax.fill(ang, v, color=CL[m], alpha=0.08)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(metrics, fontsize=8)
ax.set_yticklabels([]); ax.set_title("Stylistic fingerprints (clean), min-max normalized", fontsize=10)
ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.25, 1.1))
save(fig, "09_style_radar.png")

# 10) first-sentence change rate -------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(corr)); w = 0.2
for k, m in enumerate(MOD):
    y = [ds[(ds.model == m) & (ds.condition == c)]["first_sentence_change_rate"].iloc[0] for c in corr]
    ax.bar(x + (k - 1.5) * w, y, w, color=CL[m], label=LAB[m])
ax.set_xticks(x); ax.set_xticklabels(corr); ax.set_ylim(0, 1)
ax.set_ylabel("P(first sentence changes)")
ax.set_title("The OPENING sentence is highly unstable under corruption "
             "(even for the most content-stable model)", fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
save(fig, "10_first_sentence_change.png")

print("\nAll figures written to outputs/figures/")
