#!/usr/bin/env python3
"""make_decoupling_multimodel.py — the utility half of Table 1, paired with its safety half.

WHAT THIS IS FOR. fig:decoupling is the paper's central claim (safety and capability
are separable properties) and it currently rests on ONE checkpoint: ten corruptions
applied to the safety-tuned LLaVA-CoT. Part 14 measured ScienceQA-250 accuracy for
all six Table-1 configurations under the same three corruptions, so the same scatter
can now be drawn with 18 points spanning six models instead of 10 points spanning one.

TWO MODES, deliberately split so the numbers can be read before a plot exists:
  (default)  accuracy table + Pearson r + the zero-utility-cost cell census.
             STDLIB ONLY -- safe to run on a Newton login node (still prefix with
             OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1).
  --fig      additionally redraws REU_WACV/figures/decoupling_multimodel.pdf.
             Imports matplotlib, so run it on the Mac, never on a login node.

    python make_decoupling_multimodel.py --dir ~/experiments/part14/results
    python make_decoupling_multimodel.py --dir ../../results_newton/part14 --fig

PROVENANCE OF THE SAFETY HALF. The HR_C values below are transcribed from
tab:siuo-main in sec/4_results.tex. They are NOT recomputed here: the part4 judged
artifacts that produced them no longer exist on disk (no results_part4/, no
part4_hr_summary.csv, no part4 judge cache -- only the 24 raw response jsonls
survive). Re-judging those 20 cells would let this script read per-item verdicts
instead of a hardcoded table, and would additionally unlock flip_flow_analysis.py on
the headline claim. Until that happens, treat this dict as a transcription that must
be re-checked by eye against the .tex whenever the table changes.
"""
import argparse
import glob
import json
import os

CORRUPTIONS = ["clean", "zoom_blur", "snow", "glass_blur"]

# model_key (part14 --model) -> (paper row label, {condition: HR_C from tab:siuo-main})
SAFETY = {
    "base_llama":           ("Llama-3.2-V (base)",     {"clean": 60.5, "glass_blur": 61.1, "snow": 64.7, "zoom_blur": 69.5}),
    "llava_cot":            ("LLaVA-CoT",              {"clean": 68.3, "glass_blur": 71.3, "snow": 71.9, "zoom_blur": 74.8}),
    "qwen2_5_vl":           ("Qwen2.5-VL",             {"clean": 70.1, "glass_blur": 73.0, "snow": 73.0, "zoom_blur": 74.2}),
    "llamav_o1":            ("LlamaV-o1",              {"clean": 83.2, "glass_blur": 80.8, "snow": 82.0, "zoom_blur": 86.8}),
    "r1_onevision":         ("R1-Onevision",           {"clean": 83.8, "glass_blur": 85.6, "snow": 84.4, "zoom_blur": 86.8}),
    "r1_onevision_nothink": ("R1-Onevision (no-think)", {"clean": 83.8, "glass_blur": 90.4, "snow": 89.8, "zoom_blur": 88.6}),
}

# The two cells tab:siuo-main greys out: non-significant DECREASES under exact
# McNemar (p=0.56, p=0.82). Kept in the scatter but drawn hollow, because dropping
# inconvenient negative cells would be exactly the cherry-pick a reviewer looks for.
NONSIG = {("llamav_o1", "glass_blur"), ("llamav_o1", "snow")}

EXPECTED_N = 250


def load_cell(path):
    """-> (n_lines, n_correct, n_unjudged). Fails loudly rather than averaging garbage."""
    n = ncorrect = nunjudged = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            v = rec.get("correct")
            if v is None:          # judge could not parse a CORRECT/INCORRECT verdict
                nunjudged += 1
            elif v:
                ncorrect += 1
    return n, ncorrect, nunjudged


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir of sqa_<corruption>_<model>.jsonl")
    ap.add_argument("--fig", action="store_true", help="also redraw the PDF (needs matplotlib)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report cells with <250 items instead of aborting (progress checks)")
    args = ap.parse_args()

    d = os.path.expanduser(args.dir)
    acc, missing, partial = {}, [], []
    for m in SAFETY:
        for c in CORRUPTIONS:
            p = os.path.join(d, "sqa_%s_%s.jsonl" % (c, m))
            if not os.path.exists(p):
                missing.append((m, c))
                continue
            n, ncor, nunj = load_cell(p)
            if n != EXPECTED_N:
                partial.append((m, c, n))
                if not args.allow_partial:
                    continue
            if n == 0:
                continue
            # Unjudged items are excluded from the denominator, but a cell where the
            # grader failed on a large share is not an accuracy estimate at all.
            if nunj > 0.10 * n:
                raise SystemExit(
                    "ABORT %s/%s: judge returned None on %d of %d items (>10%%). "
                    "Accuracy would be meaningless -- inspect the grader output."
                    % (m, c, nunj, n))
            denom = n - nunj
            acc[(m, c)] = (ncor / denom * 100.0, n, nunj)

    if missing:
        print("MISSING %d of %d cells:" % (len(missing), len(SAFETY) * len(CORRUPTIONS)))
        for m, c in missing:
            print("   %-24s %s" % (m, c))
    if partial:
        print("PARTIAL (expected %d items):" % EXPECTED_N)
        for m, c, n in partial:
            print("   %-24s %-11s %d items%s"
                  % (m, c, n, "" if args.allow_partial else "  [EXCLUDED]"))
    if missing or partial:
        print()

    print("=" * 86)
    print("ScienceQA-250 accuracy (%) under ImageNet-C            [utility half, part14]")
    print("=" * 86)
    print("%-26s %8s %10s %8s %11s" % ("model", "clean", "zoom_blur", "snow", "glass_blur"))
    for m, (label, _) in SAFETY.items():
        row = "%-26s" % label
        for c in CORRUPTIONS:
            row += " %8s" % ("%.1f" % acc[(m, c)][0] if (m, c) in acc else "--")
        print(row)

    print()
    print("=" * 86)
    print("PAIRED CELLS: does safety degrade where capability does not?")
    print("=" * 86)
    print("%-26s %-11s %9s %9s %s" % ("model", "corruption", "d_acc", "d_HR_C", "note"))

    pts, zero_cost, all_cells = [], [], []
    for m, (label, hrc) in SAFETY.items():
        if (m, "clean") not in acc:
            continue
        base_acc = acc[(m, "clean")][0]
        for c in CORRUPTIONS[1:]:
            if (m, c) not in acc:
                continue
            da = acc[(m, c)][0] - base_acc
            dh = hrc[c] - hrc["clean"]
            note = "non-sig (grey in tab:siuo-main)" if (m, c) in NONSIG else ""
            if da >= 0:
                note = ("no utility cost; " + note).strip("; ")
                zero_cost.append((label, c, da, dh))
            pts.append((label, c, da, dh, (m, c) in NONSIG))
            all_cells.append(dh)
            print("%-26s %-11s %+9.1f %+9.1f %s" % (label, c, da, dh, note))

    if not pts:
        raise SystemExit("\nNo paired cells yet -- no clean cell has finished.")

    print("-" * 86)
    n_up = sum(1 for p in pts if p[3] > 0)
    r = pearson([p[2] for p in pts], [p[3] for p in pts])
    print("cells: %d   safety worsened in %d   Pearson r(d_acc, d_HR_C) = %.3f"
          % (len(pts), n_up, r))
    print("""
HOW TO READ r. The single-model figure reports r=-0.25 (n.s.): the amount of ASR
gained is unrelated to the amount of accuracy lost, which is what makes the safety
damage something other than a downstream symptom of perceptual damage. If r stays
near zero across six models the claim generalises. If r goes strongly negative here,
the honest reading is that for these models safety and utility DO fall together, and
the central claim must be narrowed to the checkpoints where it holds.""")

    if zero_cost:
        print("\nZERO-UTILITY-COST CELLS (accuracy flat or better) -- the decisive ones:")
        for label, c, da, dh in sorted(zero_cost, key=lambda t: -t[3]):
            verdict = "safety still worsened" if dh > 0 else "safety also improved"
            print("   %-26s %-11s  d_acc %+5.1f  d_HR_C %+5.1f   %s" % (label, c, da, dh, verdict))
    else:
        print("\nNo zero-utility-cost cells: every corruption cost accuracy in every model.")
        print("  The single-model JPEG/frost/fog argument then has no multi-model analogue,")
        print("  and the decoupling claim must lean on the r~0 result instead.")

    if args.fig:
        draw(pts)


def draw(pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.normpath(os.path.join(HERE, "..", "REU_WACV", "figures"))
    os.makedirs(OUT, exist_ok=True)
    INK, MUTE, GRID = "#1A1A1A", "#6b6b6b", "#e5e5e5"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42})
    # Colour follows the MODEL (the entity), never the corruption or the rank, so a
    # point keeps its identity across panels. Fixed order, never cycled.
    HUES = ["#4C78A8", "#C44E52", "#4E9A5B", "#E1873C", "#8B6BB1", "#4BA3A3"]
    MARK = {"zoom_blur": "o", "snow": "s", "glass_blur": "^"}
    labels = [lab for lab, _ in SAFETY.values()]
    color = {lab: HUES[i] for i, lab in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(3.35, 2.9))
    for lab, c, da, dh, nonsig in pts:
        ax.scatter(da, dh, s=30, marker=MARK[c], zorder=3,
                   facecolor="none" if nonsig else color[lab],
                   edgecolor=color[lab] if nonsig else "white", linewidth=1.0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel(r"$\Delta$ ScienceQA accuracy (pp)  $\leftarrow$ utility lost")
    ax.set_ylabel(r"$\Delta$ SIUO HR$_\mathrm{C}$ (pp)  less safe $\rightarrow$")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Identity is never colour-alone: a legend for the models, a second for the shapes.
    from matplotlib.lines import Line2D
    lm = [Line2D([], [], marker="o", ls="", ms=4.5, color=color[l], label=l) for l in labels]
    lc = [Line2D([], [], marker=MARK[c], ls="", ms=4.5, color=MUTE,
                 label=c.replace("_", " ")) for c in CORRUPTIONS[1:]]
    leg = ax.legend(handles=lm, fontsize=5.6, loc="upper left", frameon=False,
                    handletextpad=0.3, labelspacing=0.25, borderpad=0.1)
    ax.add_artist(leg)
    ax.legend(handles=lc, fontsize=5.6, loc="lower right", frameon=False,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.1)
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, "decoupling_multimodel.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=600)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
