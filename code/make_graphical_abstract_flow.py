"""Graphical abstract (left-to-right process): data -> systems -> level 1 / level 2 -> results.
Usage: python3 -I make_graphical_abstract_flow.py <numbers.csv> <inword_stageB_long.csv> <fig_overview.pdf> <out.png>"""
import csv, sys, os, subprocess, tempfile
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

numbers_csv, inword_csv, overview_pdf, out = sys.argv[1:5]
N = {}
for r in csv.DictReader(open(numbers_csv)):
    try: N[r["macro"]] = float(r["value"])
    except ValueError: pass
vals = defaultdict(list)
for r in csv.DictReader(open(inword_csv)):
    if int(r["n_speech_frames"]) >= 20:          # all seeds; participant value = mean over words, then over seeds
        vals[(r["system"], r["participant"], r["seed"])].append(float(r["inspeech_pcc"]))
parts = sorted({p for _, p, _ in vals})
pv_ = defaultdict(list)
for (s, p, sd), v in vals.items(): pv_[(s, p)].append(np.mean(v))
l2 = {s: np.mean([np.mean(pv_[(s, p)]) for p in parts]) for s, _ in pv_}

# spectrogram strips (original, template, decoded) cropped from Fig. 1
tmp = tempfile.mkdtemp()
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", overview_pdf, os.path.join(tmp, "ov")], check=True)
im = np.asarray(Image.open(os.path.join(tmp, "ov.png")).convert("RGB")).astype(int)
sat = (im.max(2) - im.min(2)) > 40
cols = np.where(sat.mean(0) > 0.3)[0]; x0, x1 = cols.min(), cols.max()
rows = sat[:, x0:x1].mean(1) > 0.5
e = np.flatnonzero(np.diff(np.r_[0, rows.astype(int), 0]))
bands = [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a > 50]
strips = [im[a + 3:b - 3, x0 + 3:x1 - 3].astype(np.uint8) for a, b in bands]
half = strips[0].shape[1] // 2          # first two trials only, to keep the strips readable

systems = [("Linear", "NBLin", "Linear", "#9e9e9e"), ("NeuroIncept*", "NBNeu", "NeuroIncept", "#6aa84f"),
           ("ConvED-SR*", "NBCon", "ConvED-SR", "#e8913a"), ("CNN-BiLSTM", "NBBas", "CNN-BiLSTM", "#9cc3e6"),
           ("MR-CNN-BiLSTM", "NBHyb", "MR-CNN-BiLSTM", "#2c6db5")]
matplotlib.rcParams.update({"font.family": "Liberation Sans", "font.size": 5.6})
fig = plt.figure(figsize=(3.4, 1.85), dpi=400)
bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)

def box(x, y, w, h, fc="#f3f3f3"):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec="#888888", lw=0.4))
def arrow(x0, y0, x1, y1):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color="#444444"))
def head(x, y, t):
    bg.text(x, y, t, ha="center", va="bottom", fontsize=5.8, fontweight="bold")

# 1) data
head(0.12, 0.90, "1  Data")
box(0.015, 0.10, 0.21, 0.78)
bg.text(0.12, 0.83, "sEEG + audio\n10 participants\n~100 words each", ha="center", va="top", fontsize=5.2, linespacing=1.05)
ax = fig.add_axes([0.03, 0.40, 0.18, 0.15]); ax.imshow(strips[0][:, :half], aspect="auto"); ax.axis("off")
bg.text(0.12, 0.37, "original speech\n(mel spectrogram)", ha="center", va="top", fontsize=4.8, linespacing=1.0)
bg.text(0.12, 0.19, "one model per\nparticipant, 10 folds", ha="center", va="center", fontsize=4.8, linespacing=1.0, color="#333333")
arrow(0.228, 0.49, 0.262, 0.49)

# 2) systems
head(0.375, 0.90, "2  Systems")
box(0.265, 0.10, 0.22, 0.78)
bg.text(0.375, 0.84, "5 decoders (sEEG in)", ha="center", va="top", fontsize=4.8, fontweight="bold")
for i, (name, *_ , c) in enumerate(systems):
    y = 0.75 - i * 0.075
    bg.add_patch(plt.Rectangle((0.28, y - 0.022), 0.022, 0.044, fc=c, ec="black", lw=0.3))
    bg.text(0.31, y, name, ha="left", va="center", fontsize=4.9)
bg.text(0.375, 0.335, "Timing template", ha="center", va="center", fontsize=5.0, fontweight="bold")
bg.text(0.375, 0.29, "no neural input: mean\nspectrum by time since\nthe prompted word", ha="center", va="top", fontsize=4.8, linespacing=1.0)
bg.text(0.475, 0.115, "*re-impl.", ha="right", va="bottom", fontsize=4.6, color="#555555")
arrow(0.487, 0.66, 0.522, 0.70); arrow(0.487, 0.34, 0.522, 0.30)

# 3) evaluation at two levels, with results
head(0.76, 0.90, "3  Evaluation and results")
for k, (title, sub, v, t, ylim, verdict, y0) in enumerate([
        ("Level 1: whole spectrogram", "PCC over all frames",
         [N[m + "Pcc"] for _, m, _, _ in systems], N["NBTemPcc"], (0, 0.95),
         "no decoder significantly\nabove the template", 0.51),
        ("Level 2: inside words", "in-speech PCC per word",
         [l2[kk] for _, _, kk, _ in systems], l2["Timing template"], (0, 0.48),
         "MR-CNN-BiLSTM above\nthe other 4 (Holm)", 0.10)]):
    box(0.525, y0, 0.465, 0.37, fc="#eef3fa" if k else "#f3f3f3")
    bg.text(0.535, y0 + 0.345, title, ha="left", va="top", fontsize=5.2, fontweight="bold")
    bg.text(0.535, y0 + 0.285, sub, ha="left", va="top", fontsize=4.6, color="#333333")
    ax = fig.add_axes([0.825, y0 + 0.04, 0.155, 0.25]); ax.patch.set_alpha(0)
    ax.bar(range(5), v, color=[c for *_, c in systems], edgecolor="black", lw=0.25, width=0.8)
    ax.axhline(t, ls=(0, (2, 1.2)), color="black", lw=0.6)
    ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.6, 4.6)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    best = "best decoders" if k == 0 else "MR-CNN-BiLSTM"
    bg.text(0.535, y0 + 0.215, f"template {t:.2f}\n{best} {max(v):.2f}", ha="left", va="top", fontsize=4.4, linespacing=1.05)
    bg.text(0.535, y0 + 0.025, verdict, ha="left", va="bottom", fontsize=4.6, fontweight="bold", linespacing=1.0,
            color="#1f4e8c" if k else "#333333")
bg.text(0.905, 0.075, "dashed line: template", ha="center", va="top", fontsize=4.6, color="#333333")
fig.savefig(out, dpi=400)
print("L2 means:", {k: round(x, 3) for k, x in l2.items()})
