"""Regenerate Fig. 2 (per-participant models, three correlation metrics) from the released
stageB_long.csv, with the v3 system names. Usage: python3 -I fig_eval_matrix_v3.py <stageB_long.csv> <out.pdf>"""
import csv, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.linewidth": 0.8,
                            "pdf.fonttype": 42})
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
order = [("hybrid", "MR-CNN-BiLSTM", "#2c6db5"), ("baseline", "CNN-BiLSTM", "#9cc3e6"),
         ("linear", "Linear", "#9e9e9e"), ("convedsr", "ConvED-SR (re-impl.)", "#e8913a"),
         ("neuroincept", "NeuroIncept (re-impl.)", "#6aa84f")]
subs = [f"sub-{i:02d}" for i in range(1, 11)]
val = defaultdict(list)          # (model, subject, metric) -> fold values
tmpl = defaultdict(list)         # (subject, metric) -> template values per fold
for r in rows:
    for met in ("pcc", "pcc_in_speech", "residual_pcc"):
        val[(r["model"], r["subject"], met)].append(float(r[met]))
    if r["model"] == "hybrid":   # template is identical across models for the same fold
        tmpl[(r["subject"], "pcc")].append(float(r["template_pcc"]))
        tmpl[(r["subject"], "pcc_in_speech")].append(float(r["template_in_speech"]))

panels = [("pcc", "PCC (Pearson r)", (0.3, 0.92)), ("pcc_in_speech", "In-speech PCC (Pearson r)", (-0.11, 0.56)),
          ("residual_pcc", "Residual PCC (Pearson r)", (-0.27, 0.86))]
fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.6), sharex=True)
w = 0.15
offs = (np.arange(5) - 2) * (w + 0.015)
rng = np.random.default_rng(0)
low_total = defaultdict(int)
for ax, (met, lab, ylim) in zip(axes, panels):
    for si, s in enumerate(subs):
        for k, (m, name, col) in enumerate(order):
            v = np.array(val[(m, s, met)])
            x = si + offs[k]
            if met == "pcc":
                low = v < 0.3
                if low.any():
                    low_total[m] += int(low.sum())
                    ax.plot(x, 0.305, marker="v", color="#d62728", ms=3, clip_on=False)
                    ax.text(x, 0.322, str(int(low.sum())), color="#d62728", ha="center", fontsize=6)
                v = v[~low]
            if len(v):
                bp = ax.boxplot(v, positions=[x], widths=w, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=1), whiskerprops=dict(lw=0.7),
                                capprops=dict(lw=0.7), boxprops=dict(lw=0.6))
                bp["boxes"][0].set_facecolor(col)
                ax.scatter(x + rng.uniform(-w / 4, w / 4, len(v)), v, s=2, color="#333333", zorder=3, lw=0)
        if met in ("pcc", "pcc_in_speech"):
            t = np.mean(tmpl[(s, met)])
            ax.plot([si - 0.42, si + 0.42], [t, t], ls=(0, (3, 2)), color="black", lw=1.3, zorder=4)
    if met == "residual_pcc":
        ax.axhline(0, color="#777777", lw=0.8)
    ax.set_ylim(*ylim); ax.set_ylabel(lab)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    for si in range(1, 10):
        ax.axvline(si - 0.5, color="#f0f0f0", lw=0.6, zorder=0)
axes[-1].set_xticks(range(10)); axes[-1].set_xticklabels(subs); axes[-1].set_xlim(-0.55, 9.55)
handles = [Patch(facecolor=c, edgecolor="black", lw=0.6, label=n) for _, n, c in order]
handles += [Line2D([], [], ls=(0, (3, 2)), color="black", lw=1.3, label="Template (participant mean)"),
            Line2D([], [], marker="v", color="#d62728", ls="none", ms=3, label="folds with PCC < 0.3: count (not in box)")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handlelength=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6)
fig.savefig(out)
print("folds with PCC < 0.3:", dict(low_total))
