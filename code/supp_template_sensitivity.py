"""
supp_template_sensitivity.py -- is the Stage B picture robust to the two arbitrary choices of the template / in-speech analysis? (supplementary material)

  (a) MAXPH, the number of frames since the last word onset after which the template uses one common bucket: 150, 300 (main), 600 frames.
  (b) the threshold of the in-speech mask: the Otsu threshold of the main analysis, shifted by -0.25 and +0.25 standard deviations of the frame energy.
Nothing is retrained: it re-scores the saved Stage B predictions (<SEEG_OUT_DIR>/stageB/<model>_<sub>_fold<k>_preds.npz) of all 5 systems.
For each variant it recomputes, per participant-fold, the template (from the TRAIN blocks of that fold, as in the main analysis), the template PCC and in-speech PCC,
the in-speech PCC of every system and its residual PCC against the new template; participant value = mean over its folds; then the mean over participants.
Sanity check: the main variant (MAXPH 300, shift 0) must reproduce the values of the stageB/*.json files (printed as 'max abs difference').

Usage (SEEG_OUT_DIR set; stageB_subject.py, stageA_pooled.py, stageA_extra.py, template_floor_continuous.py in the same folder; needs the *_preds.npz files):
    python supp_template_sensitivity.py
Writes supp_template_sensitivity.csv (one row per variant, system) into SEEG_OUT_DIR.
"""
import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageB_subject as B
import template_floor_continuous as T
A = B.A
OUT = B.OUT
MODELS = ["hybrid", "baseline", "linear", "neuroincept", "convedsr"]
VARIANTS = [dict(name="main (MAXPH 300, Otsu)", maxph=300, delta=0.0),
            dict(name="MAXPH 150", maxph=150, delta=0.0),
            dict(name="MAXPH 600", maxph=600, delta=0.0),
            dict(name="Otsu - 0.25 SD", maxph=300, delta=-0.25),
            dict(name="Otsu + 0.25 SD", maxph=300, delta=+0.25)]


def otsu_threshold(e):
    """the threshold used by template_floor_continuous.speech_mask"""
    h, edges = np.histogram(e, bins=128)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def score(Y, P, Tm, sp):
    pb = A.pcc_bins
    ok = sp.sum() > 100
    return dict(pcc=float(pb(Y, P).mean()), template_pcc=float(pb(Y, Tm).mean()), residual_pcc=float(pb(Y - Tm, P - Tm).mean()),
                pcc_in_speech=float(pb(Y[sp], P[sp]).mean()) if ok else np.nan,
                template_in_speech=float(pb(Y[sp], Tm[sp]).mean()) if ok else np.nan, speech_frac=float(sp.mean()))


if __name__ == "__main__":
    rows, chk = [], []
    for sub in A.ALL_SUBS:
        for k in range(A.NBLK):
            B.set_fold(k)
            D, _, _ = A.load_all([sub]); d = D[sub]; tr = d["train_idx"]; spec = d["spec"]
            e = spec.mean(1); thr, sd = otsu_threshold(e), e.std()
            for mod in MODELS:
                f = os.path.join(B.RES, f"{mod}_{sub}_fold{k}_preds.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f); idx, P = z[sub + "_idx"], z[sub + "_P"]; Y = spec[idx]
                for v in VARIANTS:
                    T.MAXPH = v["maxph"]
                    ph, _ = T.phase_since_onset(d["labels"])
                    Tm = T.template_predict(ph[tr], spec[tr], ph[idx])
                    sp = (e > thr + v["delta"] * sd)[idx]
                    r = score(Y, P, Tm, sp)
                    rows.append(dict(variant=v["name"], system=mod, subject=sub, fold=k, **r))
                    if v["maxph"] == 300 and v["delta"] == 0.0:                       # compare with the stored main result
                        m = json.load(open(os.path.join(B.RES, f"{mod}_{sub}_fold{k}.json")))["per_subject"][sub]
                        chk.append(max(abs(r[x] - m[x]) for x in ["pcc", "template_pcc", "residual_pcc", "pcc_in_speech", "template_in_speech"]))
        print(sub, "done", flush=True)
    T.MAXPH = 300
    df = pd.DataFrame(rows)
    print(f"\nsanity check, main variant vs stored stageB/*.json: max abs difference {max(chk):.2e} over {len(chk)} participant-folds (should be ~1e-6)")
    part = df.groupby(["variant", "system", "subject"])[["pcc", "pcc_in_speech", "residual_pcc", "template_pcc", "template_in_speech", "speech_frac"]].mean().reset_index()
    res = part.groupby(["variant", "system"])[["pcc", "pcc_in_speech", "residual_pcc", "template_pcc", "template_in_speech", "speech_frac"]].mean().round(3).reset_index()
    order = [v["name"] for v in VARIANTS]
    res["variant"] = pd.Categorical(res.variant, order, ordered=True); res = res.sort_values(["variant", "system"])
    res.to_csv(os.path.join(OUT, "supp_template_sensitivity.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nmean over the 10 participants:\n", res.to_string(index=False))
    print("\nHybrid minus Baseline in-speech PCC, and Hybrid minus template in-speech PCC, per variant (participants where Hybrid is higher):")
    for v in order:
        p = part[part.variant == v].pivot(index="subject", columns="system", values="pcc_in_speech")
        t = part[(part.variant == v) & (part.system == "hybrid")].set_index("subject")["template_in_speech"]
        print(f"{v:24s} Hybrid-Baseline {np.nanmean(p.hybrid - p.baseline):+.3f} ({int((p.hybrid > p.baseline).sum())}/10) | "
              f"Hybrid-template {np.nanmean(p.hybrid - t):+.3f} ({int((p.hybrid > t).sum())}/10) | speech fraction {part[part.variant == v].speech_frac.mean():.3f}")
