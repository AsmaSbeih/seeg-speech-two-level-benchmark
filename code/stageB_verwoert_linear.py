"""
stageB_verwoert_linear.py -- Verwoert-style linear baseline on the NEW continuous 80-bin features.

Protocol (Verwoert et al. reconstruction_minimal.py, applied to our features):
  * per participant, KFold(10, shuffle=False) over contiguous frames (no nesting, no shuffling)
  * context stacking: frames t-20, t-15, ..., t+20 (order 4, step 5) -> 9 x n_channels features
    (the first/last 20 frames are dropped so that no padding is invented)
  * z-score with train-fold statistics, PCA(50) fitted on the train fold, LinearRegression to all 80 mel bins
  * PCC per mel bin over the test-fold frames, mean over bins, mean over folds   (== ConvED-SR's PCC definition)

Three variants are reported side by side so that nothing is hidden:
  real            : literal protocol
  real_purged     : same, but 40 frames (~0.46 s) next to each test fold are removed from the training set,
                    because stacked inputs of train frames at the boundary overlap with test inputs
  shifted_control : inputs circularly shifted by 12000 frames (~139 s) = misaligned with the mel; PCC must be ~0

Usage (same terminal where SEEG_OUT_DIR is set):
    python stageB_verwoert_linear.py quick     # sub-02, sub-06, sub-10 only  (feasibility test)
    python stageB_verwoert_linear.py           # all 10 participants (slower)
Needs: pip install scikit-learn
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

OUT = os.environ["SEEG_OUT_DIR"]
ORDER, STEP = 4, 5
CTX = ORDER * STEP
N_PCA = 50
SHIFT = 12000


def stack(feat):
    T = len(feat)
    idx = np.arange(CTX, T - CTX)
    X = np.concatenate([feat[idx + k * STEP] for k in range(-ORDER, ORDER + 1)], axis=1).astype(np.float32)
    return X, idx


def pcc_per_bin(y, p):
    y = y - y.mean(0); p = p - p.mean(0)
    return (y * p).sum(0) / (np.sqrt((y ** 2).sum(0) * (p ** 2).sum(0)) + 1e-12)


def run(X, Y, purge):
    res = []
    for tr, te in KFold(10, shuffle=False).split(X):
        if purge:
            tr = tr[(tr < te[0] - purge) | (tr > te[-1] + purge)]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        pca = PCA(N_PCA).fit((X[tr] - mu) / sd)
        lr = LinearRegression().fit(pca.transform((X[tr] - mu) / sd), Y[tr])
        P = lr.predict(pca.transform((X[te] - mu) / sd))
        res.append(pcc_per_bin(Y[te], P))
    return np.array(res)                                   # (10 folds, 80 bins)


if __name__ == "__main__":
    quick = sys.argv[1:] == ["quick"]
    subs = ["sub-02", "sub-06", "sub-10"] if quick else [f"sub-{i:02d}" for i in range(1, 11)]
    rows, bins = [], {}
    for sub in subs:
        t0 = time.time()
        feat = np.load(os.path.join(OUT, f"{sub}_feat.npy")); spec = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
        X, idx = stack(feat); Y = spec[idx]
        r_real = run(X, Y, 0)
        r_pur = run(X, Y, 2 * CTX)
        r_sh = run(np.roll(X, SHIFT, axis=0), Y, 0)
        bins[sub] = r_real.mean(0)
        rows.append(dict(subject=sub, channels=feat.shape[1], real=r_real.mean(), real_purged=r_pur.mean(),
                         shifted_control=r_sh.mean(), fold_min=r_real.mean(1).min(), fold_max=r_real.mean(1).max()))
        print(f"{sub}: real {r_real.mean():.3f} | purged {r_pur.mean():.3f} | shifted control {r_sh.mean():+.3f} "
              f"| folds [{r_real.mean(1).min():.3f}, {r_real.mean(1).max():.3f}]  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print("\nmean over participants:\n", df[["real", "real_purged", "shifted_control"]].mean().round(3).to_string())
    tag = "quick" if quick else "all"
    df.to_csv(os.path.join(OUT, f"stageB_verwoert_linear_{tag}.csv"), index=False)
    pd.DataFrame(bins).to_csv(os.path.join(OUT, f"stageB_verwoert_linear_perbin_{tag}.csv"), index_label="mel_bin")
    print("saved csv files in", OUT)
