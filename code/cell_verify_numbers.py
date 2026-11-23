# ===== VERIFY: recompute the paper's numbers from the RAW result files (independent of my CSV and scripts) =====
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
OUT = os.environ["SEEG_OUT_DIR"]                 # set SEEG_OUT_DIR before running (folder with the extracted features)

EXPECTED = {'NATemPccM': 0.744,
 'NATemInsM': 0.236,
 'NAHybPccM': 0.696,
 'NAHybInsM': 0.232,
 'NAHybResM': 0.369,
 'NAHybPccS': 0.004,
 'NABasPccM': 0.705,
 'NABasInsM': 0.219,
 'NAConPccM': 0.59,
 'NANeuPccM': 0.427,
 'NALinPccM': 0.472,
 'NANullHybPcc': 0.17,
 'NANullBasPcc': 0.044,
 'NAHBPccP': 0.322,
 'NAHBPccWin': 3.0,
 'NATHybPccP': 0.105,
 'NATHybInsP': 0.625,
 'NBTemPcc': 0.714,
 'NBTemIns': 0.214,
 'NBHybPcc': 0.752,
 'NBHybIns': 0.3,
 'NBHybRes': 0.477,
 'NBBasPcc': 0.754,
 'NBBasIns': 0.27,
 'NBLinPcc': 0.675,
 'NBConPcc': 0.623,
 'NBNeuPcc': 0.556,
 'NBHBasInsP': 0.006,
 'NBHBasInsWin': 9.0,
 'NBTHybInsP': 0.037,
 'NBTHybInsWin': 7.0,
 'NBNeuLow': 25.0,
 'NBConLow': 1.0,
 'NBHybBeatT': 62.0,
 'NBHybMin': 0.404,
 'NPvSHybPccA': 0.696,
 'NPvSHybPccB': 0.758,
 'NPvSHybPccP': 0.002,
 'NPvSHybPccWin': 10.0,
 'NPvSHybInsA': 0.232,
 'NPvSHybInsB': 0.287,
 'NPvSHybInsP': 0.064,
 'NPvSHybRatio': 92.0,
 'NAStoiCst': 0.512,
 'NBStoiCst': 0.502,
 'NAStoiTru': 0.718,
 'NAStoiHyb': 0.337,
 'NAStoiTem': 0.372,
 'NBStoiHyb': 0.383,
 'NBStoiTem': 0.324,
 'NBEstoiHyb': 0.025,
 'NAEstoiCst': -0.003,
 'NAMcdHyb': 31.5,
 'NBMcdHyb': 31.0}

# ---------- raw loading (own code) ----------
def load(dirname, stage_b=False):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, dirname, "*.json"))):
        j = json.load(open(p))
        if "per_subject" not in j: continue
        if stage_b and "fold" not in j: continue
        for sub, r in j["per_subject"].items():
            rows.append(dict(model=j["model"], lr=j.get("lr", np.nan), seed=j.get("seed", -1), subject=sub, fold=j.get("fold", -1), **r))
    return pd.DataFrame(rows)

A = load("stageA"); B = load("stageB", stage_b=True)
LR = {"baseline": 1e-4, "hybrid": 1e-4, "neuroincept": 1e-3, "convedsr": 1e-3}
def selA(m):
    d = A[(A.model == m) & (A.lr.sub(LR[m]).abs() < 1e-9) & (A.seed.isin([42, 123, 456]))]
    return d
def seedmean(m, k):
    return selA(m).groupby("seed")[k].apply(lambda s: s.mean()).values       # mean over participants, per seed
def part(m, k):                                                            # per participant, mean over seeds
    return selA(m).groupby("subject")[k].mean().sort_index().values
lin = A[A.model == "linear"]
def W(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    return float(wilcoxon(d).pvalue), int((d > 0).sum())

R = {}   # recomputed values
hybA = selA("hybrid")
R["NATemPccM"] = hybA.template_pcc.mean(); R["NATemInsM"] = hybA.template_in_speech.mean()
for tag, m in [("Hyb", "hybrid"), ("Bas", "baseline"), ("Con", "convedsr"), ("Neu", "neuroincept")]:
    R[f"NA{tag}PccM"] = seedmean(m, "pcc").mean()
R["NAHybInsM"] = seedmean("hybrid", "pcc_in_speech").mean(); R["NAHybResM"] = seedmean("hybrid", "residual_pcc").mean()
R["NAHybPccS"] = seedmean("hybrid", "pcc").std(ddof=1); R["NABasInsM"] = seedmean("baseline", "pcc_in_speech").mean()
R["NALinPccM"] = lin.pcc.mean()
R["NANullHybPcc"] = A[A.model == "hybrid_null"].pcc.mean(); R["NANullBasPcc"] = A[A.model == "baseline_null"].pcc.mean()
R["NAHBPccP"], R["NAHBPccWin"] = W(part("hybrid", "pcc"), part("baseline", "pcc"))
tp = hybA.groupby("subject")[["template_pcc", "template_in_speech"]].mean().sort_index()
R["NATHybPccP"], _ = W(part("hybrid", "pcc"), tp.template_pcc.values)
R["NATHybInsP"], _ = W(part("hybrid", "pcc_in_speech"), tp.template_in_speech.values)

hB = B[B.model == "hybrid"]
R["NBTemPcc"] = hB.template_pcc.mean(); R["NBTemIns"] = hB.template_in_speech.mean()
for tag, m in [("Hyb", "hybrid"), ("Bas", "baseline"), ("Lin", "linear"), ("Con", "convedsr"), ("Neu", "neuroincept")]:
    R[f"NB{tag}Pcc"] = B[B.model == m].pcc.mean()
R["NBHybIns"] = hB.pcc_in_speech.mean(); R["NBHybRes"] = hB.residual_pcc.mean(); R["NBBasIns"] = B[B.model == "baseline"].pcc_in_speech.mean()
R["NBNeuLow"] = int((B[B.model == "neuroincept"].pcc < 0.3).sum()); R["NBConLow"] = int((B[B.model == "convedsr"].pcc < 0.3).sum())
R["NBHybBeatT"] = int((hB.pcc > hB.template_pcc).sum()); R["NBHybMin"] = hB.pcc.min()
pB = lambda m, k: B[B.model == m].groupby("subject")[k].mean().sort_index().values
R["NBHBasInsP"], R["NBHBasInsWin"] = W(pB("hybrid", "pcc_in_speech"), pB("baseline", "pcc_in_speech"))
tB = hB.groupby("subject")[["template_in_speech"]].mean().sort_index()
R["NBTHybInsP"], R["NBTHybInsWin"] = W(pB("hybrid", "pcc_in_speech"), tB.template_in_speech.values)

# pooled (Stage A) vs subject-dependent (Stage B) on the SAME test blocks: Stage A TEST_B=(2,7) = Stage B folds 2 and 7
xa = {k: part("hybrid", k) for k in ["pcc", "pcc_in_speech"]}
xb = {k: B[(B.model == "hybrid") & (B.fold.isin([2, 7]))].groupby("subject")[k].mean().sort_index().values for k in ["pcc", "pcc_in_speech"]}
R["NPvSHybPccA"] = xa["pcc"].mean(); R["NPvSHybPccB"] = xb["pcc"].mean(); R["NPvSHybRatio"] = 100 * xa["pcc"].mean() / xb["pcc"].mean()
R["NPvSHybPccP"], R["NPvSHybPccWin"] = W(xb["pcc"], xa["pcc"])
R["NPvSHybInsA"] = xa["pcc_in_speech"].mean(); R["NPvSHybInsB"] = xb["pcc_in_speech"].mean(); R["NPvSHybInsP"], _ = W(xb["pcc_in_speech"], xa["pcc_in_speech"])

# ---------- audio files (raw) ----------
def find(name):
    hits = glob.glob(os.path.join(OUT, "**", name), recursive=True)
    return hits[0] if hits else None
def jl(name):
    p = find(name); print("  using", p) if p else print("  NOT FOUND:", name)
    return pd.DataFrame(json.load(open(p)).values()) if p else pd.DataFrame()
SA, SB, MC = jl("stoi_results.json"), jl("stoiB_results.json"), jl("mcd_results.json")
CA, CB = jl("stageA_constant_control.json"), jl("stoiB_constant_control.json")
sysm = lambda df, s, k="stoi": df[df.system == s][k].mean()
R["NAStoiTru"] = sysm(SA, "truth"); R["NAStoiHyb"] = sysm(SA, "hybrid"); R["NAStoiTem"] = sysm(SA, "template")
R["NBStoiHyb"] = sysm(SB, "hybrid"); R["NBStoiTem"] = sysm(SB, "template"); R["NBEstoiHyb"] = sysm(SB, "hybrid", "estoi")
if len(CA): R["NAStoiCst"] = CA.stoi.mean(); R["NAEstoiCst"] = CA.estoi.mean()
if len(CB): R["NBStoiCst"] = CB.stoi.mean()
if len(MC):
    mm = MC.groupby(["stage", "system", "subject"]).mcd.mean().reset_index().groupby(["stage", "system"]).mcd.mean()
    R["NAMcdHyb"] = mm[("A", "hybrid")]; R["NBMcdHyb"] = mm[("B", "hybrid")]

# ---------- compare ----------
print("raw rows: Stage A", len(A), "| Stage B", len(B), "| models A", sorted(A.model.unique()), "| models B", sorted(B.model.unique()))
print("\nStage B rows per (model, lr, seed) - expect 100 each:"); print(B.groupby(["model", B.lr.fillna(-1), "seed"]).size().to_string())
print("\nStage A rows per (model, lr, seed) - expect 10 each:"); print(A.groupby(["model", A.lr.fillna(-1), "seed"]).size().to_string())
bad = []; print(f"\n{'item':14s} {'paper':>9s} {'recomputed':>11s}  status")
for k, e in EXPECTED.items():
    if k not in R: print(f"{k:14s} {e:9.3f} {'--':>11s}  MISSING (file not found)"); bad.append(k); continue
    a = float(R[k]); tol = 0.051 if "Mcd" in k else (0.51 if k.endswith("Ratio") else 0.0006)
    ok = abs(a - e) <= tol
    print(f"{k:14s} {e:9.3f} {a:11.3f}  {'OK' if ok else '<<<<< DIFFERENT'}"); 
    if not ok: bad.append(k)
print(f"\n{len(EXPECTED) - len(bad)} of {len(EXPECTED)} match." + ("  ALL MATCH." if not bad else f"  Differences: {bad}"))
