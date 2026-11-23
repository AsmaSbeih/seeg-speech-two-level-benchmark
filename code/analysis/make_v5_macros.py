"""Regenerate every macro of the v5 manuscript that is not produced by make_numbers.py.
Families: NBT*Holm (level-1 Holm family), NBTemShareBas, NBT*Lo/Hi (level-1 bootstrap), NW* (level 2, seed 42 and
seeds 123/456/mean), NLRA*/NLRB* (both levels without the folds whose test blocks were used for learning-rate selection).

Usage: python3 make_v5_macros.py stageB_long.csv inword_stageB_long_2.csv [out.csv] [out.tex]
Writes rows "macro,value,source" (stdout or out.csv) and optionally \\newcommand lines (out.tex) to append to numbers.*.
Conventions: per-participant means over folds (level 1) or words (level 2, words with >= 20 speech frames);
exact two-sided Wilcoxon signed-rank test by full enumeration (n = 10); Holm step-down; seed 42 unless stated."""
import csv, sys, itertools
from collections import defaultdict
import numpy as np

L1, L2 = sys.argv[1], sys.argv[2]
OUT = []
def put(m, v, src): OUT.append((m, v, src))
f3 = lambda x: f"{x:.3f}"; s3 = lambda x: f"{x:+.3f}"; f4 = lambda x: f"{x:.4f}"

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
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = 0
    for r, i in enumerate(o): m = max(m, min(1, (len(ps) - r) * ps[i])); out[i] = m
    return out
M = lambda x: sum(x) / len(x)

# ---------- level 1 (stageB_long.csv, seed 42) ----------
L1rows = [r for r in csv.DictReader(open(L1)) if r.get("seed", "42") == "42"]
DEC1 = [("linear", "Lin"), ("baseline", "Bas"), ("hybrid", "Hyb"), ("neuroincept", "Neu"), ("convedsr", "Con")]
def l1(excl=()):
    v = defaultdict(list)
    for r in L1rows:
        if int(r["fold"]) + 1 in excl: continue          # file folds 0-9 = paper folds 1-10
        s, m = r["subject"], r["model"]
        v[("pcc", m, s)].append(float(r["pcc"])); v[("ins", m, s)].append(float(r["pcc_in_speech"]))
        if m == "baseline":
            v[("pcc", "template", s)].append(float(r["template_pcc"])); v[("ins", "template", s)].append(float(r["template_in_speech"]))
    subs = sorted({k[2] for k in v})
    return {(met, m): np.array([M(v[(met, m, s)]) for s in subs]) for (met, m, _) in v}
P1 = l1()
fam = [(met, m, a) for met in ("pcc", "ins") for m, a in DEC1]
ph = holm([wil(list(P1[(met, m)] - P1[(met, "template")])) for met, m, a in fam])
for (met, m, a), h in zip(fam, ph):
    put(f"NBT{a}{'Pcc' if met == 'pcc' else 'Ins'}Holm", f3(h), "Holm over the 10 template comparisons (5 decoders x PCC, in-speech PCC), seed 42")
put("NBTemShareBas", str(int(round(100 * P1[("pcc", "template")].mean() / P1[("pcc", "baseline")].mean()))), "100 x template PCC / CNN-BiLSTM PCC")
idx = np.random.default_rng(2026).integers(0, 10, size=(10000, 10))
for m, a in DEC1:
    if m == "hybrid": continue
    for met, tag in (("pcc", "Pcc"), ("ins", "Ins")):
        d = P1[(met, m)] - P1[(met, "template")]; lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        put(f"NBT{a}{tag}Lo", s3(lo), "bootstrap 95% (default_rng(2026), 10,000 resamples of participants), decoder minus template")
        put(f"NBT{a}{tag}Hi", s3(hi), "bootstrap 95%, decoder minus template")

# ---------- level 2 (inword file) ----------
W = list(csv.DictReader(open(L2)))
DEC2 = [("MR-CNN-BiLSTM", "Hyb"), ("CNN-BiLSTM", "Bas"), ("Linear", "Lin"), ("NeuroIncept", "Neu"), ("ConvED-SR", "Con")]
def l2(seed="42", excl=()):
    v = defaultdict(list)
    for r in W:
        sys_ = r["system"]; sd = "42" if sys_ in ("Linear", "Timing template") else seed
        if r["seed"] != sd or int(r["n_speech_frames"]) < 20 or int(r["fold"]) + 1 in excl: continue
        v[(sys_, r["participant"])].append(float(r["inspeech_pcc"]))
    parts = sorted({p for _, p in v})
    return {s: np.array([M(v[(s, p)]) for p in parts]) for s, _ in DEC2 + [("Timing template", "Tem")]}, \
           sum(len(x) for (s, p), x in v.items() if s == "MR-CNN-BiLSTM")
def compare(P):
    best = max([s for s, _ in DEC2], key=lambda s: P[s].mean())
    oth = [(s, a) for s, a in DEC2 if s != best]
    D = [P[best] - P[s] for s, _ in oth]; ps = [wil(list(d)) for d in D]
    return best, oth, D, ps, holm(ps)
w42 = [r for r in W if r["seed"] == "42" and r["system"] == "MR-CNN-BiLSTM"]
put("NWWordsFold", str(len(w42)), "words with all speech frames inside one test fold")
P, nw = l2()
put("NWWords", str(nw), "words kept (>= 20 speech frames)"); put("NWWordsShort", str(len(w42) - nw), "words with < 20 speech frames")
for s, a in DEC2 + [("Timing template", "Tem")]: put(f"NW{a}InsMean", f3(P[s].mean()), "mean over participants of the per-participant mean word-level in-speech PCC")
best, oth, D, ps, hs = compare(P)
assert best == "MR-CNN-BiLSTM", best
for (s, a), d, p, h in zip(oth, D, ps, hs):
    put(f"NWH{a}InsD", s3(d.mean()), f"{best} minus {s}"); put(f"NWH{a}InsWin", str(int((d > 0).sum())), "participants higher")
    put(f"NWH{a}InsP", f4(p), "exact Wilcoxon"); put(f"NWH{a}InsHolm", f4(h), "Holm over the 4 comparisons")
d = P[best] - P["Timing template"]
put("NWHTemInsD", s3(d.mean()), "descriptive, best minus template"); put("NWHTemInsWin", str(int((d > 0).sum())), "descriptive")
om = [P[s].mean() for s, _ in oth]
put("NWOthInsMin", f3(min(om)), "min mean of the other decoders"); put("NWOthInsMax", f3(max(om)), "max mean of the other decoders")
put("NWHInsMaxHolm", f4(max(hs)), "max Holm"); put("NWHInsMinWin", str(min(int((x > 0).sum()) for x in D)), "min participants higher")
def seedvals(seed=None):
    if seed: return l2(seed)[0]
    Ps = [l2(s)[0] for s in ("42", "123", "456")]
    return {k: np.mean([Q[k] for Q in Ps], axis=0) for k in Ps[0]}
for tag, seed in (("SdB", "123"), ("SdC", "456"), ("SdM", None)):
    b, oth_, D_, ps_, hs_ = compare(seedvals(seed))
    for (s, a), d, h in zip(oth_, D_, hs_):
        put(f"NWH{a}InsD{tag}", s3(d.mean()), f"seed {seed or 'mean of 42/123/456'}; highest mean: {b}")
        put(f"NWH{a}InsWin{tag}", str(int((d > 0).sum())), f"seed {seed or 'mean'}")
        put(f"NWH{a}InsHolm{tag}", f4(h), f"seed {seed or 'mean'}")

# ---------- learning-rate selection overlap ----------
for tag, ex in (("A", (1, 6)), ("B", (2, 7))):
    Q = l1(ex)
    put(f"NLR{tag}TemPcc", f3(Q[("pcc", "template")].mean()), f"without paper folds {ex}")
    put(f"NLR{tag}BasPcc", f3(Q[("pcc", "baseline")].mean()), f"without paper folds {ex}")
    put(f"NLR{tag}HybPcc", f3(Q[("pcc", "hybrid")].mean()), f"without paper folds {ex}")
    put(f"NLR{tag}MinHolm", f3(min(holm([wil(list(Q[(met, m)] - Q[(met, "template")])) for met, m, a in fam]))), f"without paper folds {ex}")
    P2, nw2 = l2(excl=ex); b, oth_, D_, ps_, hs_ = compare(P2)
    put(f"NLR{tag}Words", str(nw2), f"without paper folds {ex}"); put(f"NLR{tag}HybIns", f3(P2[b].mean()), f"without paper folds {ex}")
    put(f"NLR{tag}DMin", s3(min(d.mean() for d in D_)), f"without paper folds {ex}"); put(f"NLR{tag}DMax", s3(max(d.mean() for d in D_)), f"without paper folds {ex}")
    put(f"NLR{tag}MaxHolm", f4(max(hs_)), f"without paper folds {ex}")

out_csv = open(sys.argv[3], "w") if len(sys.argv) > 3 else sys.stdout
w = csv.writer(out_csv, lineterminator="\n"); w.writerow(["macro", "value", "source"]); w.writerows(OUT)
if len(sys.argv) > 4:
    with open(sys.argv[4], "w") as f:
        for m, v, _ in OUT: f.write("\\newcommand{\\%s}{%s}\n" % (m, v.replace("-", "\\ensuremath{-}", 1) if v.startswith("-") else v))
