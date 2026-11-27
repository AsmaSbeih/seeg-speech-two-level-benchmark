"""make_numbers_v12.py -- every number of the main text and of the supplement that depends on the decoders.
Participant value = mean over the 10 folds (level 1) or over the kept words (level 2) of one seed, then mean over seeds 42/123/456
(Linear and the template are deterministic: seed 42 only). Exact two-sided Wilcoxon (full enumeration, n = 10), Holm step-down,
bootstrap over participants with default_rng(2026), 10,000 resamples.
Usage: python make_numbers_v12.py <SEEG_OUT_DIR or the repository's results folder> <out_prefix>
       (set CONVED_HG_DIR to the folder of the HG-frame ConvED-SR runs for Supplementary S5; in the repository: results/runs/convedsr_hgframes)
Writes <out_prefix>_macros.tex / _macros.csv (every decoder-dependent macro of the paper) and <out_prefix>_stageB_long_allseeds.csv."""
import json, glob, os, sys, itertools
import numpy as np, pandas as pd
R, OUTP = sys.argv[1], sys.argv[2]
def RP(*parts):                       # works on SEEG_OUT_DIR (stageB/, quality/) and on the repository layout (runs/stageB/, files at the top)
    for cand in (os.path.join(R, *parts), os.path.join(R, "runs", *parts), os.path.join(R, parts[-1])):
        if glob.glob(cand): return cand
    return os.path.join(R, *parts)
N = {}
def put(k, v, src): N[k] = (v, src)
f3 = lambda x: f"{x:.3f}".replace("-0.000", "0.000")
s3 = lambda x: (f"{x:+.3f}" if abs(x) >= 0.0005 else "0.000")
def fp(p): return f"{p:.3f}" if p >= 0.001 else "<0.001"
f4 = lambda p: f"{p:.4f}"
def wil(d):
    d = [x for x in d if x != 0]; n = len(d); a = [abs(x) for x in d]; o = sorted(range(n), key=lambda i: a[i]); rk = [0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and a[o[j+1]] == a[o[i]]: j += 1
        for k in range(i, j+1): rk[o[k]] = (i+j)/2+1
        i = j+1
    tot = sum(rk); wp = sum(r for r, x in zip(rk, d) if x > 0); obs = min(wp, tot-wp)
    return sum(min(w, tot-w) <= obs+1e-12 for w in (sum(r for r, b in zip(rk, s) if b) for s in itertools.product([0, 1], repeat=n))) / 2**n
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0]*len(ps); m = 0
    for r, i in enumerate(o): m = max(m, min(1, (len(ps)-r)*ps[i])); out[i] = m
    return out
IDX = np.random.default_rng(2026).integers(0, 10, size=(10000, 10))
def ci(d): lo, hi = np.percentile(np.asarray(d)[IDX].mean(1), [2.5, 97.5]); return lo, hi

# ---------------- level 1: long table of participant-folds ----------------
rows = []
for f in glob.glob(RP("stageB", "*.json")) + glob.glob(RP("stageB_seeds_preds", "*.json")):
    j = json.load(open(f)); s = j["subject"]; r = j["per_subject"][s]
    rows.append(dict(model=j["model"], seed=int(j.get("seed") or 42), subject=s, fold=int(j["fold"]), lr=j.get("lr"), **r))
L = pd.DataFrame(rows)
L = L[L.model.isin(["linear", "baseline", "hybrid", "neuroincept", "convedsr"])]
L.to_csv(OUTP + "_stageB_long_allseeds.csv", index=False)
cnt = L.groupby(["model", "seed"]).size(); assert (cnt == 100).all(), cnt
SUBS = sorted(L.subject.unique()); assert len(SUBS) == 10
MOD = [("linear", "Lin"), ("baseline", "Bas"), ("hybrid", "Hyb"), ("neuroincept", "Neu"), ("convedsr", "Con")]
MET = [("pcc", "Pcc"), ("pcc_in_speech", "Ins"), ("residual_pcc", "Res"), ("mse", "Mse")]
ps_ = L.groupby(["model", "seed", "subject"])[[m for m, _ in MET] + ["template_pcc", "template_in_speech"]].mean()
def pv(model, met, seed=None):          # participant values (10,), seed-averaged unless a seed is given
    d = ps_.loc[model][met].unstack(0)   # subject x seed
    if seed is not None: return d[seed].loc[SUBS].values
    return d.loc[SUBS].mean(axis=1).values
T = {"pcc": ps_.loc[("hybrid", 42)]["template_pcc"].loc[SUBS].values, "pcc_in_speech": ps_.loc[("hybrid", 42)]["template_in_speech"].loc[SUBS].values}
# the template is the same for every model and seed
for m, _ in MOD:
    for sd in L[L.model == m].seed.unique():
        assert np.allclose(ps_.loc[(m, sd)]["template_pcc"].loc[SUBS].values, T["pcc"], atol=1e-6)
SRC = "mean over participants of seed-averaged participant values (seeds 42/123/456; Linear deterministic)"
put("NBTemPcc", f3(T["pcc"].mean()), "template, mean over participants"); put("NBTemIns", f3(T["pcc_in_speech"].mean()), "template in speech")
for m, a in MOD:
    for k, t in MET:
        v = pv(m, k).mean(); put(f"NB{a}{t}", f"{v:.2f}" if k == "mse" else f3(v), SRC)
    d = L[L.model == m]; nf = len(d)
    put(f"NB{a}Low", str(int((d.pcc < 0.3).sum())), f"participant-folds with PCC < 0.3, of {nf} (all seeds)")
    put(f"NB{a}NFolds", str(nf), "participant-folds over all seeds")
    put(f"NB{a}Min", f3(d.pcc.min()), "lowest fold PCC, all seeds")
    put(f"NB{a}LowSubjects", str(int((d.groupby('subject').pcc.min() < 0.3).sum())), "participants with at least one fold PCC < 0.3")
put("NBTemShareBas", str(int(round(100*T["pcc"].mean()/max(pv("baseline","pcc").mean(), pv("hybrid","pcc").mean())))), "100 x template PCC / best decoder PCC")
fam = [(k, t, m, a) for k, t in MET[:2] for m, a in MOD]
P = [wil(list(pv(m, k) - T[k])) for k, t, m, a in fam]; H = holm(P)
for (k, t, m, a), p, h in zip(fam, P, H):
    d = pv(m, k) - T[k]
    put(f"NBT{a}{t}D", s3(d.mean()), "decoder minus template"); put(f"NBT{a}{t}Win", str(int((d > 0).sum())), "participants above template")
    put(f"NBT{a}{t}P", fp(p), "exact Wilcoxon"); put(f"NBT{a}{t}Holm", fp(h), "Holm over 10 template comparisons")
    lo, hi = ci(d); put(f"NBT{a}{t}Lo", s3(lo), "bootstrap 2.5%"); put(f"NBT{a}{t}Hi", s3(hi), "bootstrap 97.5%")
for m, a in MOD:
    lo, hi = ci(pv(m, "residual_pcc")); put(f"NB{a}ResLo", f3(lo), "bootstrap 2.5% residual PCC"); put(f"NB{a}ResHi", f3(hi), "bootstrap 97.5%")
    for k, t in MET[:3]:
        lo, hi = ci(pv(m, k)); put(f"NB{a}{t}CiLo", f3(lo), "bootstrap mean 2.5%"); put(f"NB{a}{t}CiHi", f3(hi), "bootstrap mean 97.5%")
for k, t in MET[:2]:
    lo, hi = ci(T[k]); put(f"NBTem{t}CiLo", f3(lo), "bootstrap"); put(f"NBTem{t}CiHi", f3(hi), "bootstrap")
# MR-CNN-BiLSTM versus each comparator (12 tests, Holm), seed-averaged
famH = [(m, a, k, t) for m, a in MOD if m != "hybrid" for k, t in MET[:3]]
PH = [wil(list(pv("hybrid", k) - pv(m, k))) for m, a, k, t in famH]; HH = holm(PH)
for (m, a, k, t), p, h in zip(famH, PH, HH):
    d = pv("hybrid", k) - pv(m, k)
    put(f"NBH{a}{t}D", s3(d.mean()), "MR-CNN-BiLSTM minus comparator, seed-averaged"); put(f"NBH{a}{t}Win", str(int((d > 0).sum())), "participants")
    put(f"NBH{a}{t}P", fp(p), "exact Wilcoxon"); put(f"NBH{a}{t}Holm", fp(h), "Holm over 12")
    lo, hi = ci(d); put(f"NBH{a}{t}Lo", s3(lo), "bootstrap"); put(f"NBH{a}{t}Hi", s3(hi), "bootstrap")
# per seed (supplement S1)
for m, a in MOD[1:]:
    for k, t in MET[:3]:
        per = [pv(m, k, sd).mean() for sd in (42, 123, 456)]
        put(f"NS{a}{t}M", f3(np.mean(per)), "mean over seeds of the participant mean"); put(f"NS{a}{t}S", f3(np.std(per, ddof=1)), "SD over seeds")
        for sd, v in zip(("A", "B", "C"), per): put(f"NS{a}{t}Sd{sd}", f3(v), "seed 42/123/456")
for sd, tag in ((42, "A"), (123, "B"), (456, "C")):
    famS = [(m, a, k, t) for m, a in MOD if m != "hybrid" for k, t in MET[:3]]
    def y(m, k): return pv(m, k) if m == "linear" else pv(m, k, sd)
    PS = [wil(list(pv("hybrid", k, sd) - y(m, k))) for m, a, k, t in famS]; HS = holm(PS)
    for (m, a, k, t), h in zip(famS, HS): put(f"NBH{a}{t}HolmSd{tag}", fp(h), f"seed {sd}")



sig = [float(N[f"NBH{a}{t}HolmSd{tg}"][0].replace("<", "")) for tg in "ABC" for m, a in MOD if m not in ("hybrid",) for k, t in MET[:3] if not (m == "baseline" and k != "pcc_in_speech")]
ns = [float(N[f"NBHBas{t}HolmSd{tg}"][0]) for tg in "ABC" for t in ("Pcc", "Res")]
put("NBHSigMaxHolmSeeds", fp(max(sig)), "max Holm over seeds of the comparisons that are significant in the seed mean")
put("NBHBasNsMinHolmSeeds", fp(min(ns)), "min Holm over seeds, MR vs CNN-BiLSTM in PCC and residual PCC")

# ---------------- extra level-1 summaries ----------------
put("NBTMinHolm", fp(min(H)), "smallest Holm-adjusted p of the 10 template comparisons")
put("NBTBelowMinHolm", fp(min(h for (k, t, m, a), h in zip(fam, H) if k == "pcc" and m in ("linear", "neuroincept", "convedsr"))), "smallest Holm p, Linear/NeuroIncept/ConvED-SR vs template in PCC")
put("NBResLoMin", f3(min(ci(pv(m, "residual_pcc"))[0] for m, a in MOD)), "lowest lower bound of the residual-PCC bootstrap intervals")
# ---------------- controls of the reference ----------------
TS = pd.read_csv(RP("supp_template_sensitivity.csv"))
tt = TS[TS.system == "hybrid"].set_index("variant")
for v, tag in (("MAXPH 150", "CapA"), ("main (MAXPH 300, Otsu)", "CapM"), ("MAXPH 600", "CapB")):
    put(f"NRTemPcc{tag}", f3(tt.loc[v, "template_pcc"]), f"template PCC, {v}"); put(f"NRTemIns{tag}", f3(tt.loc[v, "template_in_speech"]), f"template in-speech, {v}")
SH = pd.read_csv(RP("shift_scan_all.csv"))
put("NRShiftMean", f3(SH.pcc.mean()), "linear, shifted target, mean over 120 runs"); put("NRShiftMin", f3(SH.pcc.min()), "min"); put("NRShiftMax", f3(SH.pcc.max()), "max")
put("NRShiftPos", str(int((SH.pcc > 0).sum())), "runs with PCC > 0"); put("NRShiftN", str(len(SH)), "runs")


# ---------------- supplement: channel counts, template sensitivity, ConvED-SR input ----------------
from scipy.stats import spearmanr
man = {s_: v for s_, v in zip(SUBS, [127, 127, 127, 115, 60, 127, 127, 54, 117, 122])}   # channels per participant (Supplementary Table S11)
ch = np.array([man[s_] for s_ in SUBS])
for m, a in MOD + [("template", "Tem")]:
    for k, t in MET[:2]:
        y = T[k] if m == "template" else pv(m, k)
        rho, p = spearmanr(ch, y); put(f"NCh{a}{t}Rho", s3(rho) if rho < 0 else f3(rho), "Spearman rho, channels vs score"); put(f"NCh{a}{t}P", f3(p), "Spearman p")
VAR = [("main (MAXPH 300, Otsu)", "A"), ("MAXPH 150", "B"), ("MAXPH 600", "C"), ("Otsu - 0.25 SD", "D"), ("Otsu + 0.25 SD", "E")]
for v, tv in VAR:
    d = TS[TS.variant == v].set_index("system")
    for m, a in MOD: put(f"NSens{tv}{a}", f3(d.loc[m, "pcc_in_speech"]), f"in-speech PCC, {v}, seed 42")
    put(f"NSens{tv}Tem", f3(d.loc["hybrid", "template_in_speech"]), f"template in-speech, {v}"); put(f"NSens{tv}Frac", f3(d.loc["hybrid", "speech_frac"]), "speech fraction")
put("NSensResMaxChange", f3(max(abs(TS[TS.variant == v].set_index("system").residual_pcc - TS[TS.variant == VAR[0][0]].set_index("system").residual_pcc).max() for v, _ in VAR)), "max change of residual PCC across variants")
HG = os.environ.get("CONVED_HG_DIR")          # folder with the HG-frame ConvED-SR runs (stageB/ and stageB_seeds_preds/ structure)
if HG:
    rr = []
    for f in glob.glob(os.path.join(HG, "**", "convedsr_*.json"), recursive=True):
        j = json.load(open(f)); s_ = j["subject"]; rr.append(dict(seed=int(j["seed"]), subject=s_, **j["per_subject"][s_]))
    G = pd.DataFrame(rr); assert len(G) == 300, len(G)
    g = G.groupby(["subject", "seed"])[["pcc", "pcc_in_speech", "residual_pcc"]].mean().groupby("subject").mean().loc[SUBS]
    for k, t in MET[:3]:
        put(f"NCHg{t}", f3(g[k].mean()), "ConvED-SR with HG frames, seed mean")
        d = pv("convedsr", k) - g[k].values
        put(f"NCKvsHg{t}D", s3(d.mean()), "1024 Hz minus HG frames"); put(f"NCKvsHg{t}Win", str(int((d > 0).sum())), "participants"); put(f"NCKvsHg{t}P", fp(wil(list(d))), "exact Wilcoxon")
    put("NCHgLow", str(int((G.pcc < 0.3).sum())), "HG-frame ConvED-SR folds with PCC < 0.3, of 300")

put("NXParamsConvedsr", "0.98", "trainable parameters of ConvED-SR with 1024 Hz input (977,886)")
# ---------------- level 2 ----------------
W = pd.read_csv(RP("inword_stageB_long.csv"))
DEC2 = [("MR-CNN-BiLSTM", "Hyb"), ("CNN-BiLSTM", "Bas"), ("Linear", "Lin"), ("NeuroIncept", "Neu"), ("ConvED-SR", "Con")]
w42 = W[(W.seed == 42) & (W.system == "MR-CNN-BiLSTM")]
K = W[W.n_speech_frames >= 20]
put("NWWordsFold", str(len(w42)), "words with all speech frames inside one test fold")
put("NWWords", str(int((w42.n_speech_frames >= 20).sum())), "words kept"); put("NWWordsShort", str(int((w42.n_speech_frames < 20).sum())), "excluded")
def l2(seed=None):
    out = {}
    for s, _ in DEC2 + [("Timing template", "Tem")]:
        d = K[K.system == s]
        g = d.groupby(["seed", "participant"]).inspeech_pcc.mean().unstack(0).loc[SUBS]
        out[s] = g[seed].values if (seed is not None and seed in g.columns) else (g[42].values if seed is not None else g.mean(axis=1).values)
    return out
def comp(Pw):
    best = max([s for s, _ in DEC2], key=lambda s: Pw[s].mean()); oth = [(s, a) for s, a in DEC2 if s != best]
    D = [Pw[best] - Pw[s] for s, _ in oth]; ps = [wil(list(d)) for d in D]; return best, oth, D, ps, holm(ps)
for tag, seed in (("", None), ("SdA", 42), ("SdB", 123), ("SdC", 456)):
    Pw = l2(seed); best, oth, D, ps, hs = comp(Pw); assert best == "MR-CNN-BiLSTM", (tag, best)
    for s, a in DEC2 + [("Timing template", "Tem")]: put(f"NW{a}InsMean{tag}", f3(Pw[s].mean()), "word-level in-speech PCC, mean over participants")
    for (s, a), d, p, h in zip(oth, D, ps, hs):
        put(f"NWH{a}InsD{tag}", s3(d.mean()), f"MR-CNN-BiLSTM minus {s}"); put(f"NWH{a}InsWin{tag}", str(int((d > 0).sum())), "participants")
        put(f"NWH{a}InsP{tag}", f4(p), "exact Wilcoxon"); put(f"NWH{a}InsHolm{tag}", f4(h), "Holm over 4")
        lo, hi = ci(d); put(f"NWH{a}InsLo{tag}", s3(lo), "bootstrap"); put(f"NWH{a}InsHi{tag}", s3(hi), "bootstrap")
    d = Pw["MR-CNN-BiLSTM"] - Pw["Timing template"]
    put(f"NWHTemInsD{tag}", s3(d.mean()), "descriptive"); put(f"NWHTemInsWin{tag}", str(int((d > 0).sum())), "descriptive")
    om = [Pw[s].mean() for s, _ in oth]
    put(f"NWOthInsMin{tag}", f3(min(om)), "min of other decoders"); put(f"NWOthInsMax{tag}", f3(max(om)), "max of other decoders")
    put(f"NWHInsMaxHolm{tag}", f4(max(hs)), "max Holm")
    dd = [d.mean() for (s_, a_), d in zip(oth, D) if a_ != "Bas"]
    put(f"NWHOthDMin{tag}", s3(min(dd)), "smallest MR-CNN-BiLSTM minus Linear/NeuroIncept/ConvED-SR"); put(f"NWHOthDMax{tag}", s3(max(dd)), "largest"); put(f"NWHInsMinWin{tag}", str(min(int((x > 0).sum()) for x in D)), "min participants higher")
# for the old supplement macro names (SdM = seed mean)
for k in list(N):
    if k.startswith("NWH") and not any(k.endswith(t) for t in ("SdA", "SdB", "SdC")) and k[-1] not in "ABC" and not k.endswith("Sd"):
        pass


put("NWSeedMinWin", str(min(int(N[f"NWHInsMinWin{t}"][0]) for t in ("SdA", "SdB", "SdC"))), "min participants higher over the three single-seed analyses")
put("NWSeedMaxHolm", max((N[f"NWHInsMaxHolm{t}"][0] for t in ("SdA", "SdB", "SdC")), key=float), "max Holm over the three single-seed analyses")


# all ten pairwise comparisons among the five decoders at level 2 (seed means), Holm over 10
Pw = l2(None); names = [s_ for s_, _ in DEC2]
pairs = [(a_, b_) for i, a_ in enumerate(names) for b_ in names[i + 1:]]
pp = [wil(list(Pw[a_] - Pw[b_])) for a_, b_ in pairs]; hh = holm(pp)
mr = [h for (a_, b_), h in zip(pairs, hh) if "MR-CNN-BiLSTM" in (a_, b_)]
put("NWBasSeedMinWin", str(min(int(N[f"NWHBasInsWin{t}"][0]) for t in ("SdA", "SdB", "SdC"))), "min participants MR > CNN-BiLSTM over the single-seed analyses")
put("NWAllPairsMRMaxHolm", f4(max(mr)), "max Holm (over all 10 pairwise decoder comparisons) of the four MR-CNN-BiLSTM comparisons")

# ---------------- audio (supplement; seed 42, fold 1) ----------------
S = pd.DataFrame(json.load(open(RP("quality", "stoiB_results.json"))).values())
C = pd.DataFrame(json.load(open(RP("quality", "stoiB_constant_control.json"))).values()).sort_values("subject")
for s, a in [("truth", "Tru"), ("template", "Tem")] + MOD:
    d = S[S.system == s]; assert len(d) == 10, (s, len(d)); put(f"NBStoi{a}", f3(d.stoi.mean()), "STOI, fold 1, seed 42")
put("NBStoiCst", f3(C.stoi.mean()), "constant control STOI")
h = S[S.system == "hybrid"].sort_values("subject")
put("NBHybStoiVsCstP", fp(wil(list(h.stoi.values - C.stoi.values))), "exact Wilcoxon MR-CNN-BiLSTM vs constant STOI")
put("NBHybStoiVsCstWin", str(int((h.stoi.values > C.stoi.values).sum())), "participants MR > constant")
MC = pd.DataFrame(json.load(open(RP("quality", "mcd_results.json"))).values())
MC = MC[MC.stage == "B"].groupby(["system", "subject"]).mcd.mean().groupby("system").mean()
for s, a in [("template", "Tem")] + MOD: put(f"NBMcd{a}", f"{MC[s]:.1f}", "MCD dB, seed 42, all test frames")

with open(OUTP + "_macros.tex", "w") as f:
    for k, (v, s) in sorted(N.items()):
        f.write("\\newcommand{\\%s}{%s}\n" % (k, ("\\ensuremath{-}" + v[1:] if v.startswith("-") else v).replace("<", "{<}")))
pd.DataFrame([(k, v, s) for k, (v, s) in sorted(N.items())], columns=["macro", "value", "source"]).to_csv(OUTP + "_macros.csv", index=False)
print(len(N), "macros")
