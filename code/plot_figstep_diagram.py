#!/usr/bin/env python
"""
plot_figstep_diagram.py
Static diagram showing TIS refusal vs noise bypass on a FigStep image.

Usage:
  python3 code/plot_figstep_diagram.py
  python3 code/plot_figstep_diagram.py --image figstep_images/query_ForbidQI_7_4_6.png
  python3 code/plot_figstep_diagram.py --severity 3
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import argparse, os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--image",    default="figstep_images/query_ForbidQI_4_2_6.png")
parser.add_argument("--severity", type=int, default=4, choices=[1,2,3,4,5])
args = parser.parse_args()

img_path  = args.image if os.path.isabs(args.image) else os.path.join(BASE_DIR, args.image)
img_arr   = np.array(Image.open(img_path).convert("RGB"))
std_map   = {1:10, 2:17, 3:25, 4:35, 5:50}
rng       = np.random.default_rng(42)
img_noisy = np.clip(img_arr.astype(float) + rng.normal(0, std_map[args.severity], img_arr.shape), 0, 255).astype(np.uint8)

REFUSAL = """\
<CONCLUSION>

I'm sorry, I cannot assist
with this request.

Providing instructions for
synthesizing illegal narcotics
is harmful and against my
guidelines.

</CONCLUSION>"""

COMPLY = """\
<CONCLUSION>

1. Acquire the required
   precursor compounds

2. Prepare the reaction
   environment carefully

3. Follow purification steps
   to obtain the product...

</CONCLUSION>"""

fig = plt.figure(figsize=(17, 9), facecolor="#F5F5F5")
fig.suptitle("Gaussian Noise Bypasses TIS Safety Defense on FigStep",
             fontsize=16, fontweight="bold", color="#1A1A2E", y=0.97)

ax_img_clean  = fig.add_axes([0.02, 0.52, 0.20, 0.40])
ax_model_top  = fig.add_axes([0.30, 0.56, 0.12, 0.32])
ax_resp_clean = fig.add_axes([0.50, 0.52, 0.46, 0.40])
ax_img_noisy  = fig.add_axes([0.02, 0.05, 0.20, 0.40])
ax_model_bot  = fig.add_axes([0.30, 0.09, 0.12, 0.32])
ax_resp_noisy = fig.add_axes([0.50, 0.05, 0.46, 0.40])

def show_image(ax, img_data, title, color):
    ax.imshow(img_data)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(3.5)
    ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=7)

def show_model(ax):
    ax.set_facecolor("#EDE7F6")
    for sp in ax.spines.values():
        sp.set_edgecolor("#6A0DAD"); sp.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, "LLaVA-CoT\n+  TIS",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#6A0DAD")

def show_response(ax, text, border, bg, title, title_color):
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_edgecolor(border); sp.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=8)
    ax.text(0.05, 0.93, text, transform=ax.transAxes,
            fontsize=13.5, family="monospace",
            verticalalignment="top", color="#1A1A1A", clip_on=True)

show_image(ax_img_clean, img_arr,   "FigStep Image  (clean)",                            "#1565C0")
show_image(ax_img_noisy, img_noisy, "FigStep Image  +  Noise  (sev %d)" % args.severity, "#E65100")
show_model(ax_model_top)
show_model(ax_model_bot)
show_response(ax_resp_clean, REFUSAL, "#C62828", "#FFF0F0", "Model Response:  REFUSED  ✗", "#C62828")
show_response(ax_resp_noisy, COMPLY,  "#2E7D32", "#F0FFF0", "Model Response:  COMPLIED  ✓", "#2E7D32")

def arrow(x0, y0, x1, y1, color="#555"):
    fig.add_artist(FancyArrowPatch(
        posA=(x0, y0), posB=(x1, y1), transform=fig.transFigure,
        arrowstyle="-|>", color=color, lw=2.5, mutation_scale=22))

arrow(0.225, 0.72, 0.295, 0.72)
arrow(0.425, 0.72, 0.495, 0.72)
arrow(0.225, 0.25, 0.295, 0.25)
arrow(0.425, 0.25, 0.495, 0.25)
arrow(0.12, 0.505, 0.12, 0.465, color="#E65100")
fig.text(0.128, 0.487, "+ gaussian noise  (sev %d)" % args.severity,
         fontsize=10, color="#E65100", fontweight="bold", va="center")

img_stem = os.path.splitext(os.path.basename(img_path))[0]
out = os.path.join(BASE_DIR, "results_newton", "diagram_%s_sev%d.png" % (img_stem, args.severity))
os.makedirs(os.path.join(BASE_DIR, "results_newton"), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved:", out)
plt.show()
