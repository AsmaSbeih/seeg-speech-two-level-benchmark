"""Fig. 2: structures of all models (hybrid, baseline, MR-ResBlock, NeuroIncept re-impl., ConvED-SR re-impl., linear).
All sizes are taken from stageA_pooled.py / stageA_extra.py / run_linear. Output: PDF (vector, fonts embedded) + PNG 600 dpi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})

C = dict(io="#E3E9F2", conv="#FBE5D6", mr="#DDEBD3", rnn="#EADCF0", fc="#FFF2CC", att="#D9EEF3", lin="#EDEDED")
FS = 7
fig = plt.figure(figsize=(7.16, 7.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 98); ax.axis("off")

def box(x, y, w, h, t, c, dashed=False, fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc=C[c], ec="#333333",
                                lw=0.7, ls=(0, (3, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.15)

def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", shrinkA=0, shrinkB=0, mutation_scale=6))

def title(x, y, t):
    ax.text(x, y, t, fontsize=8, fontweight="bold", ha="left", va="center")

# ---------------------------------------------------------------- (a) Hybrid / Baseline
title(0.5, 96.5, "(a) MR-CNN-BiLSTM (CNN-BiLSTM = the same network without the dashed MR-ResBlocks and with one Linear 512→80 head)")
W_, G_, H_ = 14.2, 2.8, 9.5
row1 = [("sEEG HG envelope\n127 ch × 256 frames\n(× channel mask)", "io", False),
        ("Conv1d k7, 128\nReLU, MaxPool 2", "conv", False),
        ("MR-ResBlock\n(128)", "mr", True),
        ("Conv1d k7, 256\nReLU, MaxPool 2", "conv", False),
        ("MR-ResBlock\n(256)", "mr", True)]
row2 = [("Conv1d k7, 256\nReLU, MaxPool 2", "conv", False),
        ("MR-ResBlock\n(256) × 2", "mr", True),
        ("AdaptiveAvgPool\n→ 256 frames\n+ participant emb. (32)", "io", False),
        ("BiLSTM (256 per direction)\nLinear 512→256", "rnn", False),
        ("MR-ResBlock\n(256)", "mr", True),
        ("Linear 256→80\nlog-mel\n80 × 256", "io", False)]
y1, y2 = 84.0, 70.0
for i, (t, c, d) in enumerate(row1):
    x = 0.5 + i * (W_ + G_); box(x, y1, W_, H_, t, c, d)
    if i: arr(x - G_, y1 + H_ / 2, x, y1 + H_ / 2)
for i, (t, c, d) in enumerate(row2):
    x = 0.5 + i * (W_ + G_); box(x, y2, W_, H_, t, c, d)
    if i: arr(x - G_, y2 + H_ / 2, x, y2 + H_ / 2)
xe = 0.5 + 4 * (W_ + G_) + W_
ax.plot([xe, xe + 1.5, xe + 1.5, 0.5 + W_ / 2], [y1 + H_ / 2, y1 + H_ / 2, y1 - 2.0, y1 - 2.0], color="#333", lw=0.7)
ax.plot([0.5 + W_ / 2, 0.5 + W_ / 2], [y1 - 2.0, y2 + H_], color="#333", lw=0.7)
arr(0.5 + W_ / 2, y2 + H_ + 0.01, 0.5 + W_ / 2, y2 + H_)
ax.text(99.5, 66.3, "MR-CNN-BiLSTM: 7.10 M parameters; CNN-BiLSTM: 1.96 M parameters. Input and output are both 86 frames/s (one mel frame per neural frame).",
        fontsize=6.6, ha="right", va="center", style="italic")

# ---------------------------------------------------------------- lower panels, 4 columns
top = 61.0
colx = [0.5, 25.5, 50.5, 75.5]; cw = 23.5

# (b) MR-ResBlock
x0 = colx[0]
title(x0, top, "(b) MR-ResBlock")
bw = 7.0
ax.text(x0 + cw / 2, top - 4.5, "input x (C channels)", fontsize=FS, ha="center", va="center")
arr(x0 + cw / 2, top - 6, x0 + cw / 2, top - 8.2)
for i, k in enumerate((3, 5, 7)):
    bx = x0 + 0.4 + i * (bw + 0.9)
    box(bx, top - 21.5, bw, 12.5, f"Conv1d\nk{k}\nBN\nLeaky\nReLU", "conv", fs=6.6)
    arr(x0 + cw / 2, top - 8.2, bx + bw / 2, top - 9.0)
    arr(bx + bw / 2, top - 21.5, x0 + cw / 2, top - 23.6)
box(x0 + 2, top - 29.8, cw - 4, 6.2, "Concat (3C)", "lin")
arr(x0 + cw / 2, top - 29.8, x0 + cw / 2, top - 31.6)
box(x0 + 2, top - 38.4, cw - 4, 6.8, "Conv1d 1×1 → C\nBN, LeakyReLU", "conv")
arr(x0 + cw / 2, top - 38.4, x0 + cw / 2, top - 40.2)
box(x0 + 2, top - 46.0, cw - 4, 5.8, "Dropout 0.1", "lin")
ax.add_patch(Circle((x0 + cw / 2, top - 51.6), 1.6, fc="white", ec="#333", lw=0.7))
ax.text(x0 + cw / 2, top - 51.6, "+", fontsize=8, ha="center", va="center")
arr(x0 + cw / 2, top - 46.0, x0 + cw / 2, top - 50.0)
ax.plot([x0 + cw / 2, x0 + 0.2, x0 + 0.2], [top - 7.2, top - 7.2, top - 51.6], color="#333", lw=0.7)
arr(x0 + 0.2, top - 51.6, x0 + cw / 2 - 1.6, top - 51.6)
ax.text(x0 + cw / 2, top - 55.2, "output (C channels)", fontsize=FS, ha="center", va="center")

# (c) NeuroIncept re-implementation
x0 = colx[1]
title(x0, top, "(c) NeuroIncept (re-impl.)")
ax.text(x0 + cw, top - 1.9, "4.67 M parameters", fontsize=6.6, ha="right", style="italic")
items = [("9 context frames\n(−20…+20, step 5)\n127 ch × 9", "io"),
         ("Inception (64 × 4)\n1×1, 3, 5, MaxPool", "att"),
         ("GRU 128 → GRU 256\n→ GRU 512", "rnn"),
         ("last state → Inception\n(128 × 4) → 512", "att"),
         ("+ participant emb. (32)", "io"),
         ("Dense 1024-1024-512\n-256-128 (ReLU)", "fc"),
         ("Linear 128→80\none mel frame", "io")]
hh, gap = 5.6, 1.7; y = top - 5.0 - hh
for i, (t, c) in enumerate(items):
    box(x0 + 1, y, cw - 2, hh, t, c)
    if i < len(items) - 1: arr(x0 + cw / 2, y, x0 + cw / 2, y - gap)
    y -= hh + gap

# (d) ConvED-SR-style re-implementation
x0 = colx[2]
title(x0, top, "(d) ConvED-SR (re-impl.)")
ax.text(x0 + cw, top - 1.9, "0.91 M parameters", fontsize=6.6, ha="right", style="italic")
items = [("40-frame segment\n127 ch × 40 (7 per window)", "io"),
         ("Conv1d k3 s2, 256 → ReLU", "conv"),
         ("7 × down-sampling residual\nblocks (2-D, stride 2)", "conv"),
         ("Conv2d → latent 256\n+ participant emb. (linear)", "io"),
         ("ConvTranspose2d (2×5)", "conv"),
         ("4 × up-sampling residual\nblock + CBAM", "att"),
         ("Scale outputs: 5×10, 10×20,\n20×40, 40×80 (coarse → fine)", "fc"),
         ("final scale 40 × 80\n= 40 mel frames", "io")]
hh, gap = 4.9, 1.45; y = top - 5.0 - hh
for i, (t, c) in enumerate(items):
    box(x0 + 1, y, cw - 2, hh, t, c, fs=6.6)
    if i < len(items) - 1: arr(x0 + cw / 2, y, x0 + cw / 2, y - gap)
    y -= hh + gap

# (e) Linear (Verwoert-style)
x0 = colx[3]
title(x0, top, "(e) Linear (Verwoert-style)")
items = [("9 context frames\n(−20…+20, step 5)\n127 ch × 9", "io"),
         ("z-score\n(train statistics)", "lin"),
         ("PCA → 50 components\n(fitted on train)", "lin"),
         ("+ participant one-hot (10)", "io"),
         ("Linear regression\n(least squares)", "fc"),
         ("80 log-mel bins\n(per frame)", "io")]
hh, gap = 5.6, 1.7; y = top - 5.0 - hh
for i, (t, c) in enumerate(items):
    box(x0 + 1, y, cw - 2, hh, t, c)
    if i < len(items) - 1: arr(x0 + cw / 2, y, x0 + cw / 2, y - gap)
    y -= hh + gap

# legend
lx, ly = 0.5, 1.2
for i, (k, lab) in enumerate((("io", "input / output"), ("conv", "convolution"), ("mr", "multi-res. residual"),
                              ("rnn", "recurrent"), ("att", "inception / attention"), ("fc", "dense / regression"))):
    xx = lx + i * 16.4
    ax.add_patch(FancyBboxPatch((xx, ly), 2.4, 1.8, boxstyle="round,pad=0,rounding_size=0.3", fc=C[k], ec="#333", lw=0.5))
    ax.text(xx + 3.2, ly + 0.9, lab, fontsize=6.6, va="center")

import sys
out = sys.argv[1] if len(sys.argv) > 1 else "fig_model_structures"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=600)
print("saved", out)
