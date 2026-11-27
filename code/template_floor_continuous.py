"""
template_floor_continuous.py -- what does the TRIAL RHYTHM alone explain, and what do the neural data add beyond it?

Why: the shifted control stayed clearly > 0 at every shift (shift_scan.py). Because the task repeats every ~3 s,
a shifted neural signal is still a consistent function of the speech (same phase offset in every trial), so a model can
learn it. A shifted signal is therefore NOT a valid null for this task. The valid reference is the "time-in-trial
template": predict the mel from the time elapsed since the last word onset, using NO neural input at all.

Same folds as stageB_verwoert_linear.py (KFold(10, shuffle=False)), same stacking/trimming, same PCC definition.
Per participant we report (mean over 10 folds and 80 mel bins):
  real      : Verwoert linear model (PCA 50 + LinearRegression) from the stacked HG features
  template  : mel predicted from time-since-word-onset only (train-fold mean per phase bin), no neural data
  in speech : the same real/template PCC but computed ONLY on frames where speech is present (Otsu threshold on the mean
              log-mel), i.e. without the easy part "speech vs silence"
  residual  : PCC between (Y - template) and (prediction - template): does the prediction track deviations from the
              average trial (which word / when the voice really starts), not just the rhythm?
Word onsets are taken from sub-XX_labels.npy (the 'Stimulus' words per frame): a label change INTO a word
(anything except '', '+', 'nan').

Usage:  python template_floor_continuous.py quick     (sub-02, sub-06, sub-10)      /   python template_floor_continuous.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageB_verwoert_linear import stack, pcc_per_bin, OUT, N_PCA

NONWORD = {"", "+", "nan", "None"}
MAXPH = 300                                   # frames (~3.5 s); later frames share the last bucket


def phase_since_onset(labels):
    lab = np.asarray(labels).astype(str)
    is_word = np.array([l not in NONWORD for l in lab])
    onset = is_word & np.concatenate([[True], lab[1:] != lab[:-1]])
    ph = np.full(len(lab), -1, dtype=int)
    last = None
    for i in range(len(lab)):
        if onset[i]:
            last = i
        if last is not None:
            ph[i] = min(i - last, MAXPH - 1)
    return ph + 1, int(onset.sum())            # bucket 0 = before first onset


def template_predict(ph_tr, Y_tr, ph_te):
    glob = Y_tr.mean(0)
    tab = np.tile(glob, (MAXPH + 1, 1))
    for b in np.unique(ph_tr):
        tab[b] = Y_tr[ph_tr == b].mean(0)
    return tab[ph_te]


def speech_mask(spec):
    """frames with speech: Otsu threshold on the mean log-mel (silence floor vs speech)"""
    e = spec.mean(1)
    h, edges = np.histogram(e, bins=128)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    thr = c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]
    return e > thr


def _pcc_masked(Y, P, m):
    return pcc_per_bin(Y[m], P[m]) if m.sum() > 100 else np.full(Y.shape[1], np.nan)


def run_all(X, Y, ph, sp):
    real, tmpl, resid, real_sp, tmpl_sp = [], [], [], [], []
    for tr, te in KFold(10, shuffle=False).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        pca = PCA(N_PCA).fit((X[tr] - mu) / sd)
        lr = LinearRegression().fit(pca.transform((X[tr] - mu) / sd), Y[tr])
        P = lr.predict(pca.transform((X[te] - mu) / sd))
        T = template_predict(ph[tr], Y[tr], ph[te])
        real.append(pcc_per_bin(Y[te], P)); tmpl.append(pcc_per_bin(Y[te], T)); resid.append(pcc_per_bin(Y[te] - T, P - T))
        real_sp.append(_pcc_masked(Y[te], P, sp[te])); tmpl_sp.append(_pcc_masked(Y[te], T, sp[te]))
    return [np.array(a) for a in (real, tmpl, resid, real_sp, tmpl_sp)]


if __name__ == "__main__":
    quick = sys.argv[1:] == ["quick"]
    subs = ["sub-02", "sub-06", "sub-10"] if quick else [f"sub-{i:02d}" for i in range(1, 11)]
    rows = []
    for sub in subs:
        t0 = time.time()
        feat = np.load(os.path.join(OUT, f"{sub}_feat.npy")); spec = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
        labels = np.load(os.path.join(OUT, f"{sub}_labels.npy"), allow_pickle=True)
        ph, n_on = phase_since_onset(labels)
        vals, cnt = np.unique(labels.astype(str), return_counts=True)
        top = ", ".join(f"{str(v)!r}:{c}" for v, c in sorted(zip(vals, cnt), key=lambda z: -z[1])[:3])
        X, idx = stack(feat); Y = spec[idx]
        sp = speech_mask(Y)
        real, tmpl, resid, real_sp, tmpl_sp = run_all(X, Y, ph[idx], sp)
        rows.append(dict(subject=sub, onsets=n_on, speech_frac=sp.mean(), real=real.mean(), template=tmpl.mean(),
                         real_minus_template=real.mean() - tmpl.mean(), residual_pcc=resid.mean(),
                         real_in_speech=np.nanmean(real_sp), template_in_speech=np.nanmean(tmpl_sp),
                         folds_real_gt_template=int((real.mean(1) > tmpl.mean(1)).sum())))
        print(f"{sub}: onsets {n_on} (most common labels: {top}) | real {real.mean():.3f} | template {tmpl.mean():.3f} | "
              f"real-template {real.mean() - tmpl.mean():+.3f} | residual PCC {resid.mean():+.3f} | "
              f"folds real>template {(real.mean(1) > tmpl.mean(1)).sum()}/10\n      speech frames {sp.mean():.0%}: PCC inside speech only -> real {np.nanmean(real_sp):+.3f}, template {np.nanmean(tmpl_sp):+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print("\nmean over participants:\n", df[["real", "template", "real_minus_template", "residual_pcc", "real_in_speech", "template_in_speech"]].mean().round(3).to_string())
    df.to_csv(os.path.join(OUT, f"template_floor_continuous_{'quick' if quick else 'all'}.csv"), index=False)
