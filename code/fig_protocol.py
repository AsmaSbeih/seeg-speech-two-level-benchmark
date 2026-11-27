"""Protocol figure: (a) ten contiguous blocks per participant and the role of each block in each fold;
(b) windows inside a block (training: 256 frames every 64; validation/test: 256 frames every 256; tail not scored).
Numbers follow II-B: blocks of ~30 s (~2584 frames at 86.13 frames/s), windows of 256 frames (2.97 s).
Usage: python3 -I fig_protocol.py <out.pdf>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out = sys.argv[1]
TR, VA, TE = "#e3e3e3", "#9a9a9a", "#2b2b2b"
matplotlib.rcParams.update({"font.family": "Liberation Sans", "font.size": 7})
fig = plt.figure(figsize=(3.45, 3.35))
a = fig.add_axes([0.11, 0.395, 0.86, 0.47]); b = fig.add_axes([0.11, 0.035, 0.86, 0.22])

# (a) folds x blocks
for k in range(1, 11):
    for j in range(1, 11):
        c = TE if j == k else VA if j == (k % 10) + 1 else TR
        a.add_patch(Rectangle((j - 1 + 0.04, 10 - k + 0.08), 0.92, 0.84, fc=c, ec="none"))
a.set_xlim(0, 10); a.set_ylim(0, 10)
a.set_xticks([j - 0.5 for j in range(1, 11)], [str(j) for j in range(1, 11)])
a.set_yticks([10 - k + 0.5 for k in range(1, 11)], [str(k) for k in range(1, 11)])
a.tick_params(length=0, pad=2)
for sp in a.spines.values(): sp.set_visible(False)
a.set_xlabel("Block of the recording (about 30 s each)", labelpad=2)
a.set_ylabel("Fold", labelpad=2)
a.xaxis.set_label_position("top"); a.xaxis.tick_top()
fig.text(0.01, 0.985, "(a) Ten contiguous blocks per participant; one model per participant and fold",
         ha="left", va="top", fontsize=6.6)
hs = [Rectangle((0, 0), 1, 1, fc=c) for c in (TE, VA, TR)]
a.legend(hs, ["test", "validation", "training"], ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.02),
         frameon=False, handlelength=1.2, columnspacing=1.2, fontsize=6.8)

# (b) windows inside one block (frames)
L, W = 2584, 256
b.add_patch(Rectangle((0, 0), L, 0.5, fc="white", ec="#555555", lw=0.6))
for i in range(10):                       # validation / test windows, back to back
    b.add_patch(Rectangle((i * W, 0.06), W, 0.38, fc=TE if i % 2 == 0 else "#5a5a5a", ec="white", lw=0.6))
b.add_patch(Rectangle((10 * W, 0.06), L - 10 * W, 0.38, fc="white", ec="#555555", lw=0.4, hatch="////"))
b.annotate("not scored (<1%)", xy=(L - 12, 0.04), xytext=(L, -0.2), ha="right", va="top", fontsize=6.2,
           arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
b.text(-60, 0.25, "test and\nvalidation", ha="right", va="center", fontsize=6.4)
for i in range(6):                        # training windows, overlapping (every 64 frames)
    y = 0.65 + i * 0.16
    b.add_patch(Rectangle((i * 64, y), W, 0.11, fc=TR, ec="#777777", lw=0.4))
b.text(6 * 64 + W + 40, 0.65 + 2.5 * 0.16, "overlapping windows,\none every 64 frames", ha="left", va="center", fontsize=6.4)
b.text(-60, 0.65 + 2.5 * 0.16 + 0.04, "training", ha="right", va="center", fontsize=6.4)
b.text(W / 2, -0.18, "256 frames (2.97 s)", ha="left", va="top", fontsize=6.2)
b.annotate("", xy=(0, -0.08), xytext=(W, -0.08), arrowprops=dict(arrowstyle="<->", lw=0.5))
b.set_xlim(-420, L + 20); b.set_ylim(-0.45, 1.7); b.axis("off")
fig.text(0.01, 0.285, "(b) Windows inside one block; no window crosses a block boundary", ha="left", va="top", fontsize=6.6)
fig.savefig(out)
