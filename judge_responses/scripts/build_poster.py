#!/usr/bin/env python3
"""build_poster.py — the 48x36 conference poster, built from the paper's numbers.

WHY A SCRIPT AND NOT POWERPOINT. The previous poster was a PosterPresentations
template: its numbers were retyped by hand (so they drifted from the paper), its
figures were screenshots, and half its panels were two lines of text in a box the
size of a chart. Building it in code means every value below traces to a table in
the paper or a summary CSV, and a re-run after new results is free.

STORY (the order a reader walks it, left to right, top to bottom):
    HOOK    one image: same model, same prompt, blur alone turns REFUSE into COMPLY
    1  safety breaks before capability      <- the central claim
    2  it is not one model                  <- 16 of 18 cells, + a 476-item replication
    3  the harm-location principle          <- explains contradictions in prior work
    4  reasoning is not a safeguard         <- matched Instruct/Thinking pair
    5  where the decision lives             <- module ablation
    6  what actually helps                  <- and what does not transfer
    TAKEAWAY

COLOR. Palette slots come from the dataviz reference palette and were validated
with its checker (categorical slots 1-3 all-pairs: worst CVD dE 9.2, normal-vision
24.0; diverging poles blue/red: CVD dE 23.8). The previous poster encoded
"safer vs less safe" as RED vs GREEN, which is the textbook red-green CVD failure;
that comparison is now blue vs red, and every bar is also direct-labelled so the
encoding never rests on hue alone.

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
OUT = os.path.join(ROOT, "Poster_CharlesLevy_v2")

W, H = 48.0, 36.0                      # inches

# ---- design tokens ---------------------------------------------------------
GOLD    = "#FFC904"     # UCF brand, headers only -- never a data mark
INK     = "#111111"
MUTED   = "#5A5A57"
PANEL   = "#F4F4F1"
RULE    = "#DDDCD6"
BLUE    = "#2a78d6"     # categorical slot 1
ORANGE  = "#eb6834"     # categorical slot 2
AQUA    = "#1baf7a"     # categorical slot 3
RED     = "#d03b3b"     # diverging pole: less safe
NEUTRAL = "#9AA0A6"     # baseline / no-treatment series

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

FS_TITLE, FS_AUTH, FS_HEAD = 82, 40, 37
FS_BODY, FS_SMALL, FS_TICK = 27, 22, 22
FS_TAKE = 26

fig = plt.figure(figsize=(W, H), dpi=100)
fig.patch.set_facecolor("white")


def X(x): return x / W
def Y(y): return y / H
def ax_at(x, y, w, h, **kw):
    """axes from inches, measuring y from the TOP of the poster."""
    a = fig.add_axes([X(x), Y(H - y - h), X(w), Y(h)], **kw)
    a.set_facecolor("none")
    a.set_zorder(6)          # must beat the panel boxes appended to fig.patches
    return a


def box(x, y, w, h, fc=PANEL, ec=RULE, lw=1.6, r=0.10):
    fig.patches.append(FancyBboxPatch(
        (X(x), Y(H - y - h)), X(w), Y(h),
        boxstyle="round,pad=0,rounding_size=%f" % (r / W),
        fc=fc, ec=ec, lw=lw, transform=fig.transFigure, zorder=1))


def text(x, y, s, size=FS_BODY, weight="normal", color=INK,
         ha="left", va="top", style="normal", **kw):
    return fig.text(X(x), Y(H - y), s, fontsize=size, fontweight=weight,
                    color=color, ha=ha, va=va, style=style, zorder=5, **kw)


def header(x, y, w, num, title, h=1.02):
    """Gold section header bar with a numbered badge."""
    fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(w), Y(h),
                                 fc=GOLD, ec="none", transform=fig.transFigure, zorder=2))
    fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(0.86), Y(h),
                                 fc=INK, ec="none", transform=fig.transFigure, zorder=3))
    fig.text(X(x + 0.43), Y(H - y - h / 2), str(num), fontsize=FS_HEAD, fontweight="bold",
             color=GOLD, ha="center", va="center", zorder=4)
    fig.text(X(x + 1.15), Y(H - y - h / 2), title, fontsize=FS_HEAD, fontweight="bold",
             color=INK, ha="left", va="center", zorder=4)


def takeaway(x, y, w, s, size=FS_TAKE):
    """One-sentence 'what you should remember' strip under a panel."""
    h = 0.92
    fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(w), Y(h),
                                 fc="#FFF6D6", ec="none", transform=fig.transFigure, zorder=2))
    fig.patches.append(Rectangle((X(x), Y(H - y - h)), X(0.10), Y(h),
                                 fc=GOLD, ec="none", transform=fig.transFigure, zorder=3))
    fig.text(X(x + 0.36), Y(H - y - h / 2), s, fontsize=size, fontweight="bold",
             color=INK, ha="left", va="center", zorder=4)


def clean_axes(a, grid_axis="both"):
    a.grid(color=RULE, lw=1.1, axis=grid_axis, zorder=0)
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=FS_TICK, length=4, width=1.2)


# =============================== LAYOUT =====================================
M, GAP = 1.1, 0.55
COLW = (W - 2 * M - 2 * 0.6) / 3.0
CX = [M, M + COLW + 0.6, M + 2 * (COLW + 0.6)]

y_title, h_title = 0.70, 4.80
y_hook = y_title + h_title + 0.45;  h_hook = 8.30
y_b3   = y_hook + h_hook + 0.50;    h_b3   = 8.30
y_b4   = y_b3 + h_b3 + 0.50;        h_b4   = 8.30
y_ban  = y_b4 + h_b4 + 0.5;         h_ban  = 1.85

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
LOGO = os.path.join(os.path.dirname(HERE), "poster_assets")
for fn, lx, ly, lw_, lh in [("ucf.png", 1.3,   y_title + 1.28, 8.4, 2.35),
                            ("bu.png",  39.0,  y_title + 0.42, 5.6, 2.00),
                            ("nsf.png", 40.55, y_title + 2.62, 2.5, 2.05)]:
    pth = os.path.join(LOGO, fn)
    if os.path.exists(pth):
        a = ax_at(lx, ly, lw_, lh)
        a.imshow(mpimg.imread(pth)); a.axis("off")

# ================================ HOOK ======================================
hx = [M, M + 13.0 + 0.45, M + 13.0 + 0.45 + 18.6 + 0.45]
hw = [13.0, 18.6, 13.3]

# --- hook left: problem + question + setup
box(hx[0], y_hook, hw[0], h_hook, fc="white", ec=RULE)
text(hx[0] + 0.55, y_hook + 0.75, "THE PROBLEM", size=30, weight="bold", color=MUTED)
text(hx[0] + 0.55, y_hook + 1.62,
     "Safety alignment is tested on\n"
     "$\\bf{clean}$ images — and deployed on\n"
     "blurry, noisy, compressed ones.", size=FS_BODY)
fig.patches.append(Rectangle((X(hx[0] + 0.55), Y(H - y_hook - 5.42)), X(hw[0] - 1.1), Y(2.06),
                             fc=INK, ec="none", transform=fig.transFigure, zorder=2))
text(hx[0] + 0.95, y_hook + 3.78, "THE QUESTION", size=21, weight="bold", color=GOLD)
text(hx[0] + 0.95, y_hook + 4.58, "Does safety survive ordinary\nimage corruption?",
     size=30, weight="bold", color="white", style="italic")
text(hx[0] + 0.55, y_hook + 5.78, "HOW WE TEST IT", size=30, weight="bold", color=MUTED)
for i, (k, v) in enumerate([("6", "open VLMs, 8 configurations"),
                            ("10", "ImageNet-C corruptions × severities"),
                            ("7", "safety benchmarks + 2 utility controls"),
                            ("GPT-4o", "judges reasoning and answer separately")]):
    yy = y_hook + 6.35 + i * 0.50
    text(hx[0] + 0.62, yy, k, size=25, weight="bold", color=ORANGE)
    text(hx[0] + 2.55, yy, v, size=24)

# --- hook centre: THE FLIP (hero)
box(hx[1], y_hook, hw[1], h_hook, fc="white", ec=INK, lw=2.6)
text(hx[1] + hw[1] / 2, y_hook + 0.72,
     "Blur alone turns a refusal into compliance", size=33, weight="bold", ha="center")
flip = os.path.join(FIGS, "flip_2014.png")
if os.path.exists(flip):
    a = ax_at(hx[1] + 0.35, y_hook + 1.05, hw[1] - 0.7, h_hook - 2.25)
    a.imshow(mpimg.imread(flip)); a.axis("off")
text(hx[1] + hw[1] / 2, y_hook + h_hook - 0.72,
     "Same model. Same prompt. Only the image changed.",
     size=26, weight="bold", ha="center", va="center", color=RED)

# --- hook right: what we found
box(hx[2], y_hook, hw[2], h_hook, fc="white", ec=RULE)
text(hx[2] + 0.55, y_hook + 0.75, "WHAT WE FOUND", size=30, weight="bold", color=MUTED)
finds = [("Safety breaks before capability.",
          "Corruptions with $\\bf{zero}$ accuracy cost\nstill raise harmful rate."),
         ("Direction depends on where harm lives.",
          "Blur the $\\it{attack}$ $\\rightarrow$ safer ($-11.3$).\nBlur the $\\it{evidence}$ $\\rightarrow$ less safe ($+7.0$)."),
         ("Reasoning is not a safeguard.",
          "Reasoning-trained models start less\nsafe and degrade further."),
         ("Prompting helps — but not our part.",
          "A generic safety prompt transfers;\ncorruption-awareness does not.")]
for i, (a_, b_) in enumerate(finds):
    yy = y_hook + 1.50 + i * 1.78
    fig.patches.append(plt.Circle((X(hx[2] + 0.82), Y(H - yy - 0.16)), X(0.30),
                                  fc=GOLD, ec="none", transform=fig.transFigure, zorder=3))
    fig.text(X(hx[2] + 0.82), Y(H - yy - 0.16), str(i + 1), fontsize=25, fontweight="bold",
             ha="center", va="center", zorder=4)
    text(hx[2] + 1.42, yy - 0.12, a_, size=25, weight="bold")
    text(hx[2] + 1.42, yy + (0.72 if "\n" not in a_ else 1.30), b_, size=22, color=MUTED)

# ============================ PANEL 1: DECOUPLING ============================
header(CX[0], y_b3, COLW, 1, "SAFETY BREAKS BEFORE CAPABILITY")
box(CX[0], y_b3 + 1.02, COLW, h_b3 - 1.02, fc="white")
DEC = [("glass blur", -8.4, 5.4), ("defocus", -6.8, 3.6), ("motion blur", -4.8, 1.3),
       ("zoom blur", -4.0, 8.4), ("contrast", -3.6, 4.2), ("snow", -2.0, 4.2),
       ("elastic", -1.6, 1.9), ("JPEG", 0.0, 4.2), ("frost", 0.8, 4.2), ("fog", 1.6, 2.4)]
NUDGE = {"JPEG": (-0.55, -0.30), "frost": (0.18, 0.40), "fog": (0.15, 0.34),
         "snow": (0.18, 0.30), "contrast": (0.18, 0.30), "elastic": (0.18, 0.30),
         "zoom blur": (0.22, 0.22), "glass blur": (0.22, 0.24),
         "defocus": (0.22, 0.30), "motion blur": (0.22, 0.30)}
a = ax_at(CX[0] + 1.55, y_b3 + 1.75, COLW - 2.5, h_b3 - 4.15)
a.axhspan(0, 10, xmin=(0 + 9.6) / 12.4, xmax=1.0, color="#FFF1F1", zorder=0)
for n, du, dh in DEC:
    zero = du >= 0
    a.scatter(du, dh, s=470 if zero else 330, marker="D" if zero else "o",
              color=RED if zero else BLUE, edgecolor="white", linewidth=2.4, zorder=4)
    dx, dy = NUDGE[n]
    a.annotate(n, (du, dh), textcoords="offset points",
               xytext=(dx * 26, dy * 26), fontsize=20,
               ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
               color=RED if zero else INK, fontweight="bold" if zero else "normal")
a.axhline(0, color=INK, lw=2.0); a.axvline(0, color=INK, lw=2.0, ls=(0, (5, 4)))
a.set_xlim(-9.6, 2.8); a.set_ylim(0, 10)
a.set_xlabel(r"$\Delta$ ScienceQA accuracy (pp)     $\leftarrow$ utility lost", fontsize=25, labelpad=10)
a.set_ylabel(r"$\Delta$ SIUO harmful rate (pp)" "\n" r"less safe $\rightarrow$", fontsize=25, labelpad=10)
a.text(1.35, 9.3, "zero utility\ncost", fontsize=21, color=RED, ha="center",
       va="top", fontweight="bold")
clean_axes(a)
a.legend(handles=[Line2D([], [], marker="o", ls="", ms=17, color=BLUE, label="costs accuracy"),
                  Line2D([], [], marker="D", ls="", ms=17, color=RED, label="no accuracy cost")],
         fontsize=21, loc="lower left", frameon=False, handletextpad=0.4)
takeaway(CX[0] + 0.35, y_b3 + h_b3 - 1.30, COLW - 0.7,
         "All 10 corruptions weaken safety.\nJPEG, frost and fog cost no accuracy at all.", size=24)

# ========================= PANEL 2: NOT ONE MODEL ===========================
header(CX[1], y_b3, COLW, 2, "IT IS NOT ONE MODEL")
box(CX[1], y_b3 + 1.02, COLW, h_b3 - 1.02, fc="white")
T1 = [("Llama-3.2-V (base)", 60.5, 61.1, 64.7, 69.5),
      ("LLaVA-CoT",          68.3, 71.3, 71.9, 74.8),
      ("Qwen2.5-VL",         70.1, 73.0, 73.0, 74.2),
      ("LlamaV-o1",          83.2, 80.8, 82.0, 86.8),
      ("R1-Onevision",       83.8, 85.6, 84.4, 86.8),
      ("R1-OV (no-think)",   83.8, 90.4, 89.8, 88.6)]
tx, ty, tw = CX[1] + 0.55, y_b3 + 1.45, COLW - 1.1
colx = [tx + 5.55, tx + 7.90, tx + 10.25, tx + 12.60]
for j, c in enumerate(["clean", "glass", "snow", "zoom"]):
    text(colx[j], ty + 0.30, c, size=23, weight="bold", ha="center", va="center", color=MUTED)
text(tx, ty + 0.30, "SIUO harmful rate (%)", size=23, weight="bold", va="center", color=MUTED)
nworse = 0
for i, (name, cl, gl, sn, zm) in enumerate(T1):
    ry = ty + 0.82 + i * 0.74
    if i % 2 == 0:
        fig.patches.append(Rectangle((X(tx - 0.18), Y(H - ry - 0.42)), X(tw), Y(0.80),
                                     fc="#FAFAF8", ec="none", transform=fig.transFigure, zorder=1))
    text(tx, ry, name, size=24, va="center")
    text(colx[0], ry, "%.1f" % cl, size=25, ha="center", va="center", weight="bold")
    for j, v in enumerate([gl, sn, zm]):
        worse = v > cl
        nworse += worse
        fig.patches.append(Rectangle((X(colx[j + 1] - 0.92), Y(H - ry - 0.36)), X(1.84), Y(0.72),
                                     fc="#FBE3E3" if worse else "#DFEAF8", ec="none",
                                     transform=fig.transFigure, zorder=2))
        fig.text(X(colx[j + 1]), Y(H - ry), "%.1f %s" % (v, "▲" if worse else "▼"),
                 fontsize=24, ha="center", va="center", zorder=4,
                 color=RED if worse else BLUE, fontweight="bold")
assert nworse == 16, "expected 16 of 18 worse cells, got %d" % nworse
text(tx, ty + 5.08, "▲ worse than clean          ▼ better", size=21, color=MUTED)

takeaway(CX[1] + 0.35, y_b3 + h_b3 - 1.30, COLW - 0.7,
         "16 of 18 model × corruption cells get worse ($p = 0.0013$).\n"
         "Replicates on HoliSafe — an independent set 3× larger.", size=24)

# ======================= PANEL 3: HARM-LOCATION ==============================
header(CX[2], y_b3, COLW, 3, "EVEN THE MILDEST BLUR BREAKS IT")
box(CX[2], y_b3 + 1.02, COLW, h_b3 - 1.02, fc="white")
SEV = [0, 1, 2, 3, 4, 5]                       # 0 = clean
HR_R = [68.3, 77.3, 80.8, 79.6, 78.4, 80.8]    # part12_dose_response.csv, LLaVA-CoT
HR_C = [68.3, 71.3, 73.1, 74.3, 73.7, 70.7]
a = ax_at(CX[2] + 2.05, y_b3 + 2.22, COLW - 3.0, h_b3 - 5.45)
a.axvspan(-0.42, 0.42, color="#EFEEEA", zorder=0)
for ys, col, lab in [(HR_R, BLUE, "reasoning (HR$_R$)"), (HR_C, ORANGE, "final answer (HR$_C$)")]:
    a.plot(SEV, ys, "-o", color=col, lw=3.4, ms=15, markeredgecolor="white",
           markeredgewidth=2.6, label=lab, zorder=4)
# the whole point: clean -> the WEAKEST setting is most of the damage
a.annotate("", xy=(1, 77.3), xytext=(0, 68.3),
           arrowprops=dict(arrowstyle="-|>", lw=3.2, color=RED,
                           shrinkA=14, shrinkB=14, mutation_scale=32), zorder=5)
a.text(0.26, 87.0, "$+9.0$ at severity 1", fontsize=21.5, color=RED,
       fontweight="bold", ha="left", va="top")
a.set_xticks(SEV); a.set_xticklabels(["clean", "1", "2", "3", "4", "5"])
a.set_xlim(-0.55, 5.35); a.set_ylim(63, 88)
a.set_xlabel("zoom-blur severity", fontsize=25, labelpad=10)
a.set_ylabel("SIUO harmful rate (%)", fontsize=24, labelpad=8)
a.legend(fontsize=21, frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.6,
         loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
clean_axes(a)
text(CX[2] + 0.55, y_b3 + h_b3 - 2.05,
     "Severity 1 is close to imperceptible, yet it already carries most of the\n"
     "loss. Severities 2-5 sit on that same plateau rather than climbing.",
     size=21.5, color=MUTED)
takeaway(CX[2] + 0.35, y_b3 + h_b3 - 1.30, COLW - 0.7,
         "There is no safe amount of corruption — the damage is\ndone at the weakest setting we can apply.", size=24)

# ==================== PANEL 4: REASONING IS NOT A SAFEGUARD ==================
header(CX[0], y_b4, COLW, 4, "REASONING IS NOT A SAFEGUARD")
box(CX[0], y_b4 + 1.02, COLW, h_b4 - 1.02, fc="white")
CONDS = ["clean", "glass\nblur", "snow", "zoom\nblur"]
INS = [46.1, 46.7, 47.9, 53.3]
THK = [53.3, 60.5, 63.5, 64.7]
a = ax_at(CX[0] + 1.75, y_b4 + 2.30, COLW - 2.6, h_b4 - 5.55)
xs = range(4); bw = 0.36
a.bar([x - bw / 2 - 0.015 for x in xs], INS, bw, color=BLUE, zorder=3, label="Instruct")
a.bar([x + bw / 2 + 0.015 for x in xs], THK, bw, color=ORANGE, zorder=3, label="Thinking")
for x, (i_, t_) in enumerate(zip(INS, THK)):
    a.text(x - bw / 2, i_ + 1.4, "%.1f" % i_, ha="center", fontsize=20, color=BLUE, fontweight="bold")
    a.text(x + bw / 2, t_ + 1.4, "%.1f" % t_, ha="center", fontsize=20, color=ORANGE, fontweight="bold")
a.set_xticks(list(xs)); a.set_xticklabels(CONDS, fontsize=21)
a.set_ylim(0, 72); a.set_ylabel("SIUO harmful rate (%)", fontsize=23, labelpad=8)
a.legend(fontsize=22, frameon=False, ncol=2, handletextpad=0.4, columnspacing=2.0,
         loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
clean_axes(a, grid_axis="y")
text(CX[0] + 0.55, y_b4 + h_b4 - 2.34,
     "Qwen3-VL-8B — one family, one scale, only reasoning post-training differs.\n"
     "Thinking starts $+7.2$ less safe and loses $+11.4$ under corruption (vs $+7.2$).",
     size=21.5, color=MUTED)
takeaway(CX[0] + 0.35, y_b4 + h_b4 - 1.05, COLW - 0.7,
         "The reasoning model is worse at baseline $\\it{and}$ degrades more.", size=24)

# ====================== PANEL 5: WHERE THE DECISION LIVES ====================
header(CX[1], y_b4, COLW, 5, "WHERE THE DECISION LIVES")
box(CX[1], y_b4 + 1.02, COLW, h_b4 - 1.02, fc="white")
MOD = [("clean data\nLLM only", 72.5, 31.1), ("corruption-aware\nLLM only", 67.1, 31.7),
       ("corruption-aware\nLLM + vision", 72.5, 34.7), ("corruption-aware\nvision only", 50.3, 50.9)]
a = ax_at(CX[1] + 1.75, y_b4 + 2.30, COLW - 2.6, h_b4 - 5.55)
xs = range(4); bw = 0.36
a.bar([x - bw / 2 - 0.015 for x in xs], [m[1] for m in MOD], bw, color=BLUE, zorder=3,
      label="reasoning (HR$_R$)")
a.bar([x + bw / 2 + 0.015 for x in xs], [m[2] for m in MOD], bw, color=ORANGE, zorder=3,
      label="final answer (HR$_C$)")
for x, m in enumerate(MOD):
    a.text(x - bw / 2, m[1] + 1.5, "%.1f" % m[1], ha="center", fontsize=20, color=BLUE, fontweight="bold")
    a.text(x + bw / 2, m[2] + 1.5, "%.1f" % m[2], ha="center", fontsize=20, color=ORANGE, fontweight="bold")
a.set_xticks(list(xs)); a.set_xticklabels([m[0] for m in MOD], fontsize=18.5)
a.set_ylim(0, 80); a.set_ylabel("SIUO harmful rate (%)", fontsize=23, labelpad=8)
a.legend(fontsize=21, frameon=False, ncol=2, handletextpad=0.4, columnspacing=2.0,
         loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
clean_axes(a, grid_axis="y")
text(CX[1] + 0.55, y_b4 + h_b4 - 2.34,
     "Adapting the $\\bf{vision\\ tower}$ gives the safest reasoning in the study (50.3)\n"
     "— and the $\\bf{worst}$ answers (50.9). The decision is implemented in the LLM.",
     size=21.5, color=MUTED)
takeaway(CX[1] + 0.35, y_b4 + h_b4 - 1.05, COLW - 0.7,
         "Safer perception never reaches the answer.", size=24)

# ======================== PANEL 6: WHAT ACTUALLY HELPS =======================
header(CX[2], y_b4, COLW, 6, "WHAT ACTUALLY HELPS")
box(CX[2], y_b4 + 1.02, COLW, h_b4 - 1.02, fc="white")
PR = [("LLaVA-CoT\n(prompt built here)", 71.6, 60.7, 53.5),
      ("Qwen2.5-VL\n(held out)",         72.6, 55.6, 59.8),
      ("R1-Onevision\n(held out)",       85.2, 70.9, 78.0)]
a = ax_at(CX[2] + 1.75, y_b4 + 2.30, COLW - 2.6, h_b4 - 5.55)
xs = range(3); bw = 0.25
for k, (lbl, col) in enumerate([("no prompt", NEUTRAL), ("generic safety", BLUE),
                                ("+ corruption-aware", ORANGE)]):
    a.bar([x + (k - 1) * (bw + 0.03) for x in xs], [p[k + 1] for p in PR], bw,
          color=col, zorder=3, label=lbl)
for x, p in enumerate(PR):
    for k, v in enumerate(p[1:]):
        a.text(x + (k - 1) * (bw + 0.03), v + 1.4, "%.1f" % v, ha="center",
               fontsize=18.5, fontweight="bold",
               color=[NEUTRAL, BLUE, ORANGE][k])
a.set_xticks(list(xs)); a.set_xticklabels([p[0] for p in PR], fontsize=19.5)
a.set_ylim(0, 95); a.set_ylabel("SIUO harmful rate (%)", fontsize=23, labelpad=8)
a.legend(fontsize=20, frameon=False, ncol=3, handletextpad=0.4, columnspacing=1.4,
         loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
clean_axes(a, grid_axis="y")
for x in (1, 2):     # the two held-out models, where corruption-awareness backfires
    a.annotate("worse", xy=(x + (bw + 0.03), PR[x][3] + 10.5), fontsize=19.5,
               color=ORANGE, ha="center", fontweight="bold")
text(CX[2] + 0.55, y_b4 + h_b4 - 2.34,
     "A $\\bf{generic}$ safety prompt wins all 12 cells — free, no training. Adding\n"
     "“the image may be corrupted” wins 4/4 where built, loses 8/8 held out.",
     size=21.5, color=MUTED)
takeaway(CX[2] + 0.35, y_b4 + h_b4 - 1.05, COLW - 0.7,
         "We report this against our own proposed defense.", size=24)

# =============================== BANNER =====================================
fig.patches.append(Rectangle((0, Y(H - y_ban - h_ban)), 1, Y(h_ban),
                             fc=INK, ec="none", transform=fig.transFigure, zorder=2))
fig.patches.append(Rectangle((0, Y(H - y_ban - h_ban)), 1, Y(0.13),
                             fc=GOLD, ec="none", transform=fig.transFigure, zorder=3))
text(W / 2, y_ban + h_ban / 2 - 0.10,
     "Utility benchmarks cannot certify safety.", size=46, weight="bold",
     color="white", ha="center", va="center")
text(W / 2, y_ban + h_ban / 2 + 0.62,
     "Make image corruption a first-class axis in safety evaluation and training.",
     size=31, color=GOLD, ha="center", va="center")

# ------------------------------- FOOTER -------------------------------------
text(M, H - 0.80,
     "ImageNet-C · Hendrycks & Dietterich, ICLR 2019      SIUO · Wang et al., NAACL 2025      "
     "VLSBench · Hu et al., ACL 2025      HoliSafe · 2025      Think-in-Safety · Lou et al., 2025",
     size=19, color=MUTED, va="center")
text(W - M, H - 0.80,
     "UCF Center for Research in Computer Vision · Boston University · Supported in part by NSF",
     size=19, color=MUTED, ha="right", va="center")

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=95)
print("saved", OUT + ".pdf")
print("saved", OUT + ".png")
