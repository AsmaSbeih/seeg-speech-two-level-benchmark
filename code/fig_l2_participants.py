"""Supplementary Fig. S3: level 2, word-level in-speech PCC per participant (participant values averaged over seeds 42/123/456; words with >= 20 speech frames).
(a) participant means for the five decoders and the template, lines joining the same participant;
(b) MR-CNN-BiLSTM minus each other decoder per participant, with the Holm-adjusted p over the four comparisons.
Usage: python3 -I fig_l2_participants.py <inword_stageB_long(_2).csv> <out.pdf>"""
import csv, sys, itertools
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1:3]
v = defaultdict(list)                                  # (system, participant, seed) -> word values
for r in csv.DictReader(open(src)):
    if int(r["n_speech_frames"]) >= 20:
        v[(r["system"], r["participant"], r["seed"])].append(float(r["inspeech_pcc"]))
parts = sorted({p for _, p, _ in v})
# participant value = mean over words of one seed, then mean over the seeds present (Linear and the template: seed 42 only)
v = {(s, p): [np.mean(v[(s2, p2, sd)]) for (s2, p2, sd) in v if s2 == s and p2 == p] for (s, p, _) in list(v)}
S = [("Linear", "Linear", "#9e9e9e"), ("NeuroIncept", "Neuro-\nIncept*", "#6aa84f"), ("ConvED-SR", "ConvED-\nSR*", "#e8913a"),
     ("CNN-BiLSTM", "CNN-\nBiLSTM", "#9cc3e6"), ("MR-CNN-BiLSTM", "MR-CNN-\nBiLSTM", "#2c6db5"), ("Timing template", "Template", "#ffffff")]
P = {s: np.array([np.mean(v[(s, p)]) for p in parts]) for s, *_ in S}

def wil(d):
    d = [x for x in d if x != 0]; n = len(d); a = [abs(x) for x in d]; o = sorted(range(n), key=lambda i: a[i]); rk = [0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and a[o[j + 1]] == a[o[i]]: j += 1
        for k in range(i, j + 1): rk[o[k]] = (i + j) / 2 + 1
        i = j + 1
    tot = sum(rk); wp = sum(r for r, x in zip(rk, d) if x > 0); obs = min(wp, tot - wp)
    return sum(min(w, tot - w) <= obs + 1e-12 for w in (sum(r for r, b in zip(rk, s) if b) for s in itertools.product([0, 1], repeat=n))) / 2 ** n
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); res = [0] * len(ps); m = 0
    for r, i in enumerate(o): m = max(m, min(1, (len(ps) - r) * ps[i])); res[i] = m
    return res

matplotlib.rcParams.update({"font.family": "Liberation Sans", "font.size": 8})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.28))
x = np.arange(len(S))
for i in range(len(parts)):
    a1.plot(x, [P[s][i] for s, *_ in S], color="#c8c8c8", lw=0.6, zorder=1)
for k, (s, lab, c) in enumerate(S):
    a1.scatter(np.full(len(parts), k), P[s], s=22, color=c, edgecolor="black", lw=0.5, zorder=3)
    a1.plot([k - 0.28, k + 0.28], [P[s].mean()] * 2, color="black", lw=1.6, zorder=4)
a1.set_xticks(x, [lab for _, lab, _ in S], fontsize=7)
a1.set_ylabel("Word-level in-speech PCC\n(participant mean)")
a1.set_title("(a) Participant means", fontsize=8.5, loc="left")

oth = S[:4]
D = [P["MR-CNN-BiLSTM"] - P[s] for s, *_ in oth]
ph = holm([wil(list(d)) for d in D])
a2.axhline(0, color="black", lw=0.7)
for k, ((s, lab, c), d) in enumerate(zip(oth, D)):
    jit = np.linspace(-0.12, 0.12, len(d))
    a2.scatter(k + jit, d, s=22, color=c, edgecolor="black", lw=0.5, zorder=3)
    a2.plot([k - 0.28, k + 0.28], [d.mean()] * 2, color="black", lw=1.6, zorder=4)
    a2.text(k, max(d.max(), 0) + 0.012, f"$p_H$={ph[k]:.4f}\n{(d > 0).sum()}/10", ha="center", va="bottom", fontsize=6.5)
a2.set_xticks(range(4), [lab for _, lab, _ in oth], fontsize=7)
a2.set_ylabel("MR-CNN-BiLSTM minus decoder")
a2.set_title("(b) Paired differences", fontsize=8.5, loc="left")
lo, hi = min(d.min() for d in D), max(d.max() for d in D)
a2.set_ylim(min(lo - 0.02, -0.03), hi + 0.06)
for ax in (a1, a2):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.5); ax.grid(axis="y", color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
fig.text(0.99, 0.005, "*re-implementation. Black bar: mean over participants. Grey lines join the same participant.", ha="right", va="bottom", fontsize=6.5, color="#444444")
fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.2)
fig.savefig(out)
print({s: round(P[s].mean(), 3) for s, *_ in S}, [round(h, 4) for h in ph])
