#!/usr/bin/env python3
"""build_poster.py — the 48x36 conference poster.

STRUCTURE follows the original poster: three columns, sparse, one idea per block.
Column 1 is all text (problem -> question -> findings -> setup), column 2 carries
the two evidence charts, column 3 carries the hero image and the fix. Four visual
panels, not six -- a poster is read standing up, in about two minutes.

FIGURES are the paper's, in the paper's simple single-chart form:
  * the flip composite is the paper figure placed directly (figures/flip_9014)
  * the decoupling scatter and the dose-response curve are redrawn here from the
    same data as figures/decoupling and figures/dose_response, at poster type size

METRIC NAME. The poster says ASR throughout, matching the original poster and the
talk. The paper calls the same quantity HR_C (and HR_R for the reasoning trace),
because it scores the trace and the answer separately; where that distinction
matters here it is spelled out in words ("reasoning trace" / "final answer")
rather than with a subscript.

COLOR from the dataviz reference palette, validated with its checker: categorical
slots 1-3 pass all-pairs (worst CVD dE 9.2, normal-vision 24.0) and the diverging
poles blue/red pass at CVD dE 23.8. The original poster's safer-vs-less-safe
red/green pair is the textbook red-green CVD failure and is not used.

    python build_poster.py            # -> Poster_CharlesLevy_v2.pdf (+ .png proof)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
FIGS = os.path.join(ROOT, "REU_WACV", "figures")
LOGO = os.path.join(ROOT, "poster_assets")
OUT = os.path.join(ROOT, "Poster_CharlesLevy_v2")

W, H = 48.0, 36.0

GOLD    = "#FFC904"
INK     = "#111111"
MUTED   = "#5A5A57"
RULE    = "#DDDCD6"
BLUE    = "#2a78d6"
ORANGE  = "#eb6834"
RED     = "#d03b3b"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

FS_TITLE, FS_AUTH, FS_HEAD = 82, 40, 37

fig = plt.figure(figsize=(W, H), dpi=100)
fig.patch.set_facecolor("white")


def X(x): return x / W
def Y(y): return y / H


def ax_at(x, y, w, h):
    """Axes placed in inches, y measured from the TOP of the poster."""
    a = fig.add_axes([X(x), Y(H - y - h), X(w), Y(h)])
    a.set_facecolor("none")
    a.set_zorder(6)          # must beat the panel boxes appended to fig.patches
    return a


def box(x, y, w, h, fc="white", ec=RULE, lw=1.6, r=0.10):
    fig.patches.append(FancyBboxPatch(
        (X(x), Y(H - y - h)), X(w), Y(h),
        boxstyle="round,pad=0,rounding_size=%f" % (r / W),
        fc=fc, ec=ec, lw=lw, transform=fig.transFigure, zorder=1))


def text(x, y, s, size=27, weight="normal", color=INK, ha="left", va="top",
         style="normal", **kw):
    return fig.text(X(x), Y(H - y), s, fontsize=size, fontweight=weight, color=color,
                    ha=ha, va=va, style=style, zorder=5, **kw)


def header(x, y, w, title, num=None, h=1.02):
    fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(w), Y(h),
                                 fc=GOLD, ec="none", transform=fig.transFigure, zorder=2))
    tx = x + 0.55
    if num is not None:
        fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(0.86), Y(h),
                                     fc=INK, ec="none", transform=fig.transFigure, zorder=3))
        fig.text(X(x + 0.43), Y(H - y - h / 2), str(num), fontsize=FS_HEAD,
                 fontweight="bold", color=GOLD, ha="center", va="center", zorder=4)
        tx = x + 1.15
    fig.text(X(tx), Y(H - y - h / 2), title, fontsize=FS_HEAD, fontweight="bold",
             color=INK, ha="left", va="center", zorder=4)


def takeaway(x, y, w, s, size=26):
    h = 0.92
    fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(w), Y(h),
                                 fc="#FFF6D6", ec="none", transform=fig.transFigure, zorder=2))
    fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(0.10), Y(h),
                                 fc=GOLD, ec="none", transform=fig.transFigure, zorder=3))
    fig.text(X(x + 0.36), Y(H - y - h / 2), s, fontsize=size, fontweight="bold",
             color=INK, ha="left", va="center", zorder=4)


def place(path, x, y, w, h):
    """Drop an image into a box, preserving aspect ratio, centred."""
    img = mpimg.imread(path)
    ar = img.shape[1] / img.shape[0]
    dw, dh = (h * ar, h) if (w / h) > ar else (w, w / ar)
    a = ax_at(x + (w - dw) / 2, y + (h - dh) / 2, dw, dh)
    a.imshow(img)
    a.axis("off")


def clean_axes(a, grid_axis="both", tick=23):
    a.grid(color=RULE, lw=1.1, axis=grid_axis, zorder=0)
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=tick, length=5, width=1.3)


# =============================== LAYOUT =====================================
M = 1.1
COLW = (W - 2 * M - 2 * 0.6) / 3.0
CX = [M, M + COLW + 0.6, M + 2 * (COLW + 0.6)]

y_title, h_title = 0.70, 4.80
y_top = y_title + h_title + 0.50          # 6.00
PANEL_H = 12.40
y_row2 = y_top + PANEL_H + 0.50           # 18.90
y_ban, h_ban = 31.90, 1.85

# ------------------------------- TITLE --------------------------------------
# Follows the original poster's header: a white field with a CENTRED gold title
# panel, and the logos sitting on white to either side rather than on the gold.
TBX, TBW = 11.0, 25.4
box(TBX, y_title, TBW, h_title - 0.10, fc=GOLD, ec="none", r=0.0)
text(TBX + TBW / 2, y_title + 1.30, "Corruptions Can Weaken Safety",
     size=FS_TITLE, weight="bold", ha="center", va="center")
text(TBX + TBW / 2, y_title + 2.48, "Before They Weaken Visual Understanding",
     size=FS_TITLE, weight="bold", ha="center", va="center")
text(TBX + TBW / 2, y_title + 3.56,
     "Charles Levy$^{1}$,   Adeel Yousaf$^{2}$,   James Beetham$^{2}$",
     size=FS_AUTH, ha="center", va="center")
text(TBX + TBW / 2, y_title + 4.32,
     "$^{1}$Boston University        $^{2}$University of Central Florida",
     size=33, ha="center", va="center")

# Logos are alpha-keyed (outer white flood-filled to transparent) so they sit on
# any background without a white card around them.
for fn, lx, ly, lw_, lh in [("ucf.png", 1.3,   y_title + 1.28, 8.4, 2.35),
                            ("bu.png",  39.0,  y_title + 0.42, 5.6, 2.00),
                            ("nsf.png", 40.55, y_title + 2.62, 2.5, 2.05)]:
    p = os.path.join(LOGO, fn)
    if os.path.exists(p):
        place(p, lx, ly, lw_, lh)

# ========================= COLUMN 1 — the argument in words ==================
cx = CX[0]

header(cx, y_top, COLW, "THE PROBLEM")
box(cx, y_top + 1.02, COLW, 3.30)
text(cx + 0.55, y_top + 1.85,
     "Safety alignment is tested on\n$\\bf{clean}$ images — and deployed on\n"
     "blurry, noisy, compressed ones.", size=28)

yq = y_top + 4.75
box(cx, yq, COLW, 2.65, fc=INK, ec="none")
text(cx + 0.70, yq + 0.72, "THE QUESTION", size=22, weight="bold", color=GOLD)
text(cx + 0.70, yq + 1.55, "Does safety survive ordinary\nimage corruption?",
     size=31, weight="bold", color="white", style="italic")

yf = yq + 3.05
header(cx, yf, COLW, "WHAT WE FOUND")
box(cx, yf + 1.02, COLW, 9.85)
FINDS = [("Safety breaks before capability.",
          "Corruptions that cost $\\bf{zero}$ accuracy\nstill raise ASR. 16 of 18 cells\nworse, across 6 models."),
         ("It breaks at the mildest setting.",
          "Severity 1 — nearly invisible —\nalready does most of the damage."),
         ("Reasoning is not a safeguard.",
          "Same family, same scale: the\nreasoning model starts less safe\n$\\it{and}$ degrades further."),
         ("Direction follows where harm lives.",
          "Blur the $\\it{attack}$ $\\rightarrow$ safer ($-11.3$).\nBlur the $\\it{evidence}$ $\\rightarrow$ less safe ($+7.0$).")]
for i, (head_, body_) in enumerate(FINDS):
    yy = yf + 1.85 + i * 2.42
    fig.patches.append(plt.Circle((X(cx + 0.85), Y(H - yy - 0.14)), X(0.30),
                                  fc=GOLD, ec="none", transform=fig.transFigure, zorder=3))
    fig.text(X(cx + 0.85), Y(H - yy - 0.14), str(i + 1), fontsize=25, fontweight="bold",
             ha="center", va="center", zorder=4)
    text(cx + 1.45, yy - 0.14, head_, size=26, weight="bold")
    text(cx + 1.45, yy + 0.70, body_, size=23, color=MUTED)

yh = yf + 11.10
header(cx, yh, COLW, "HOW WE TEST IT")
box(cx, yh + 1.02, COLW, 4.15)
for i, (k, v) in enumerate([("6", "open VLMs, 8 configurations"),
                            ("10", "ImageNet-C corruptions × severities"),
                            ("7", "safety benchmarks, 2 utility controls"),
                            ("GPT-4o", "judges every response")]):
    yy = yh + 1.85 + i * 0.80
    text(cx + 0.62, yy, k, size=27, weight="bold", color=ORANGE)
    text(cx + 3.05, yy, v, size=25)

# ================== COLUMN 2 — the two evidence charts ======================
cx = CX[1]

# --- 1. decoupling. Same ten points as REU_WACV/figures/decoupling.
header(cx, y_top, COLW, "SAFETY BREAKS BEFORE CAPABILITY", num=1)
box(cx, y_top + 1.02, COLW, PANEL_H - 1.02)
DEC = [("glass blur", -8.4, 5.4), ("defocus", -6.8, 3.6), ("motion blur", -4.8, 1.3),
       ("zoom blur", -4.0, 8.4), ("contrast", -3.6, 4.2), ("snow", -2.0, 4.2),
       ("elastic", -1.6, 1.9), ("JPEG", 0.0, 4.2), ("frost", 0.8, 4.2), ("fog", 1.6, 2.4)]
NUDGE = {"JPEG": (-0.62, -0.28), "frost": (0.30, 0.42), "fog": (0.30, 0.30),
         "snow": (0.30, 0.34), "contrast": (0.30, 0.34), "elastic": (0.30, 0.34),
         "zoom blur": (0.30, 0.30), "glass blur": (0.30, 0.30),
         "defocus": (0.30, 0.34), "motion blur": (0.30, 0.34)}
a = ax_at(cx + 2.35, y_top + 2.35, COLW - 3.4, PANEL_H - 5.05)
a.axhspan(0, 10, xmin=(0 + 9.6) / 12.4, xmax=1.0, color="#FDECEC", zorder=0)
for n, du, dh in DEC:
    zero = du >= 0
    a.scatter(du, dh, s=460 if zero else 330, marker="D" if zero else "o",
              color=RED if zero else BLUE, edgecolor="white", linewidth=2.4, zorder=4)
    dx, dy = NUDGE[n]
    a.annotate(n, (du, dh), textcoords="offset points", xytext=(dx * 26, dy * 26),
               fontsize=21, ha="right" if dx < 0 else "left",
               color=RED if zero else INK, fontweight="bold" if zero else "normal")
a.axhline(0, color=INK, lw=2.0)
a.axvline(0, color=INK, lw=2.0, ls=(0, (5, 4)))
a.set_xlim(-9.6, 2.8)
a.set_ylim(0, 10)
a.set_xlabel("Δ ScienceQA accuracy (pp)      ← utility lost", fontsize=26, labelpad=10)
a.set_ylabel("Δ SIUO ASR (pp)\nless safe →", fontsize=26, labelpad=10)
a.text(1.6, 9.4, "zero utility\ncost", fontsize=22, color=RED, ha="center",
       va="top", fontweight="bold")
clean_axes(a)
a.legend(handles=[Line2D([], [], marker="o", ls="", ms=18, color=BLUE, label="costs accuracy"),
                  Line2D([], [], marker="D", ls="", ms=18, color=RED, label="no accuracy cost")],
         fontsize=22, loc="lower left", frameon=False, handletextpad=0.4)
takeaway(cx + 0.35, y_top + PANEL_H - 1.28, COLW - 0.7,
         "All 10 corruptions weaken safety. JPEG, frost\nand fog cost no accuracy at all.", size=25)

# --- 2. dose-response. Same series as REU_WACV/figures/dose_response (LLaVA-CoT).
header(cx, y_row2, COLW, "EVEN THE MILDEST BLUR BREAKS IT", num=2)
box(cx, y_row2 + 1.02, COLW, PANEL_H - 1.02)
SEV = [0, 1, 2, 3, 4, 5]
ASR_REASON = [68.3, 77.3, 80.8, 79.6, 78.4, 80.8]
ASR_ANSWER = [68.3, 71.3, 73.1, 74.3, 73.7, 70.7]
a = ax_at(cx + 2.35, y_row2 + 2.55, COLW - 3.4, PANEL_H - 5.35)
a.axvspan(-0.42, 0.42, color="#EFEEEA", zorder=0)
for ys, col, lab in [(ASR_REASON, BLUE, "reasoning trace"),
                     (ASR_ANSWER, ORANGE, "final answer")]:
    a.plot(SEV, ys, "-o", color=col, lw=3.6, ms=16, markeredgecolor="white",
           markeredgewidth=2.6, label=lab, zorder=4)
# the whole point: clean -> the WEAKEST setting is most of the damage
a.annotate("", xy=(1, 77.3), xytext=(0, 68.3),
           arrowprops=dict(arrowstyle="-|>", lw=3.4, color=RED,
                           shrinkA=15, shrinkB=15, mutation_scale=34), zorder=5)
a.text(0.30, 87.4, "$+9.0$ at severity 1", fontsize=23, color=RED,
       fontweight="bold", ha="left", va="top")
a.set_xticks(SEV)
a.set_xticklabels(["clean", "1", "2", "3", "4", "5"])
a.set_xlim(-0.55, 5.35)
a.set_ylim(63, 89)
a.set_xlabel("zoom-blur severity", fontsize=26, labelpad=10)
a.set_ylabel("SIUO ASR (%)", fontsize=26, labelpad=10)
a.legend(fontsize=23, frameon=False, ncol=2, handletextpad=0.4, columnspacing=2.0,
         loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
clean_axes(a)
takeaway(cx + 0.35, y_row2 + PANEL_H - 1.28, COLW - 0.7,
         "There is no safe amount of corruption — the\ndamage is done at the weakest setting.", size=25)

# ============= COLUMN 3 — what it looks like, and what to do ================
cx = CX[2]

# --- 3. the hero: the paper's flip composite, placed as-is
header(cx, y_top, COLW, "WHAT IT LOOKS LIKE", num=3)
box(cx, y_top + 1.02, COLW, PANEL_H - 1.02)
flip = os.path.join(FIGS, "flip_9014.png")
if os.path.exists(flip):
    place(flip, cx + 0.45, y_top + 1.35, COLW - 0.9, PANEL_H - 3.85)
text(cx + COLW / 2, y_top + PANEL_H - 2.05,
     "Same model. Same prompt. Only the image changed.",
     size=26, weight="bold", ha="center", va="center", color=RED)
takeaway(cx + 0.35, y_top + PANEL_H - 1.28, COLW - 0.7,
         "Blur alone turns a refusal into compliance.", size=25)

# --- 4. the fix, as before/after numbers rather than a chart
header(cx, y_row2, COLW, "A SIMPLE, FREE FIX", num=4)
box(cx, y_row2 + 1.02, COLW, PANEL_H - 1.02)
text(cx + 0.60, y_row2 + 1.80,
     "One line of system prompt telling the model\nto check safety. No training, no weight access.",
     size=25, color=MUTED)
text(cx + 0.60, y_row2 + 3.42, "SIUO ASR, averaged over conditions", size=22,
     weight="bold", color=MUTED)
FIX = [("LLaVA-CoT", 71.6, 60.7), ("Qwen2.5-VL", 72.6, 55.6), ("R1-Onevision", 85.2, 70.9)]
for i, (name, before, after) in enumerate(FIX):
    yy = y_row2 + 4.45 + i * 1.55
    text(cx + 0.60, yy, name, size=26, va="center")
    text(cx + 7.10, yy, "%.1f" % before, size=36, weight="bold", color=MUTED,
         ha="right", va="center")
    text(cx + 7.95, yy, r"$\rightarrow$", size=34, color=INK, ha="center", va="center")
    text(cx + 10.15, yy, "%.1f" % after, size=36, weight="bold", color=BLUE,
         ha="right", va="center")
text(cx + 0.60, y_row2 + 8.70,
     "Lower on $\\bf{all\\ 12}$ cells we tested, by 8 to 20 points.", size=25)
# the honest half: the corruption-aware variant is the part that fails to transfer
fig.patches.append(Rectangle((X(cx + 0.55), Y(H - y_row2 - PANEL_H + 1.58)), X(COLW - 1.1),
                             Y(1.34), fc="#FBEDE9", ec="none",
                             transform=fig.transFigure, zorder=2))
text(cx + 0.85, y_row2 + PANEL_H - 2.70,
     "But adding “the image may be corrupted” wins 4/4 on the\n"
     "model we built it on — and loses 8/8 on both held-out models.",
     size=22, color="#8A3B2A")
takeaway(cx + 0.35, y_row2 + PANEL_H - 1.28, COLW - 0.7,
         "Generic safety prompting transfers. Our\ncorruption-aware version does not.", size=25)

# =============================== BANNER =====================================
fig.patches.append(Rectangle((0, Y(H - y_ban - h_ban)), 1, Y(h_ban),
                             fc=INK, ec="none", transform=fig.transFigure, zorder=2))
fig.patches.append(Rectangle((0, Y(H - y_ban - h_ban)), 1, Y(0.13),
                             fc=GOLD, ec="none", transform=fig.transFigure, zorder=3))
text(W / 2, y_ban + 0.72, "Utility benchmarks cannot certify safety.",
     size=46, weight="bold", color="white", ha="center", va="center")
text(W / 2, y_ban + 1.40,
     "Make image corruption a first-class axis in safety evaluation and training.",
     size=31, color=GOLD, ha="center", va="center")

text(M, 34.65,
     "ImageNet-C · Hendrycks & Dietterich, ICLR 2019      SIUO · Wang et al., NAACL 2025      "
     "VLSBench · Hu et al., ACL 2025      HoliSafe · 2025      Think-in-Safety · Lou et al., 2025",
     size=19, color=MUTED, va="center")
text(W - M, 34.65,
     "UCF Center for Research in Computer Vision · Boston University · Supported in part by NSF",
     size=19, color=MUTED, ha="right", va="center")

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=95)
print("saved", OUT + ".pdf")
