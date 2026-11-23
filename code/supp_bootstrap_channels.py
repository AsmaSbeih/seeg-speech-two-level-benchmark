"""
supp_bootstrap_channels.py -- supplementary analyses computed from the result files only (no training, runs in seconds).

(1) Bootstrap 95% confidence intervals over the 10 participants (percentile method, 10,000 resamples, fixed seed) for the mean PCC, in-speech PCC and
    residual PCC of every system, of the template, and for the paired differences Hybrid - comparator (the same resampled participants for both).
    Participant value = mean over its folds (Stage B) or over its seeds (Stage A), as in the paper's tables.
(2) Does the number of electrodes explain the differences between participants?  Spearman correlation between the number of channels of a participant
    and its PCC / in-speech PCC (n = 10, descriptive; Stage B, per system and for the template).

Usage:   python supp_bootstrap_channels.py [results_folder]      (default: ../results next to this file, or the current folder)
Reads stageA_long.csv, stageB_long.csv, subjects_manifest.json; writes supp_bootstrap_ci.csv and supp_channels_spearman.csv in the same folder.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

D = sys.argv[1] if len(sys.argv) > 1 else (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
if not os.path.exists(os.path.join(D, "stageB_long.csv")): D = "."
KEYS = ["pcc", "pcc_in_speech", "residual_pcc"]
N_BOOT, RNG = 10000, np.random.default_rng(2026)
MODELS = ["hybrid", "baseline", "linear", "neuroincept", "convedsr"]


def participant_table(df, models):
    """rows = participants, one column per (model, metric); template added as its own 'model'"""
    out = {}
    for m in models:
        d = df[df.model == m].groupby("subject")[KEYS].mean()
        for k in KEYS: out[(m, k)] = d[k]
    t = df[df.model == "hybrid"].groupby("subject")[["template_pcc", "template_in_speech"]].mean()   # template is model-independent
    out[("template", "pcc")] = t["template_pcc"]; out[("template", "pcc_in_speech")] = t["template_in_speech"]
    return pd.DataFrame(out)


def ci(v):
    b = np.array([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    return np.percentile(b, [2.5, 97.5])


def ci_diff(a, b):
    idx = [RNG.integers(0, len(a), len(a)) for _ in range(N_BOOT)]
    bs = np.array([(a[i] - b[i]).mean() for i in idx])
    return np.percentile(bs, [2.5, 97.5])


rows = []
for stage, f in (("A", "stageA_long.csv"), ("B", "stageB_long.csv")):
    df = pd.read_csv(os.path.join(D, f))
    P = participant_table(df, MODELS)
    for (m, k) in P.columns:
        v = P[(m, k)].values
        lo, hi = ci(v)
        rows.append(dict(stage=stage, system=m, metric=k, comparison="mean", estimate=v.mean(), ci_low=lo, ci_high=hi, n_participants=len(v)))
    for m in MODELS[1:]:
        for k in KEYS:
            a, b = P[("hybrid", k)].values, P[(m, k)].values
            lo, hi = ci_diff(a, b)
            rows.append(dict(stage=stage, system=f"hybrid - {m}", metric=k, comparison="difference", estimate=(a - b).mean(), ci_low=lo, ci_high=hi, n_participants=len(a)))
    for k in ["pcc", "pcc_in_speech"]:                      # Hybrid - template
        a, b = P[("hybrid", k)].values, P[("template", k)].values
        lo, hi = ci_diff(a, b)
        rows.append(dict(stage=stage, system="hybrid - template", metric=k, comparison="difference", estimate=(a - b).mean(), ci_low=lo, ci_high=hi, n_participants=len(a)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(D, "supp_bootstrap_ci.csv"), index=False)
pd.set_option("display.width", 200)
for st in ("A", "B"):
    print(f"\n=== Stage {st}: mean over participants with 95% bootstrap CI ===")
    r = res[(res.stage == st) & (res.comparison == "mean")]
    for _, x in r.iterrows(): print(f"{x.system:12s} {x.metric:14s} {x.estimate:.3f} [{x.ci_low:.3f}, {x.ci_high:.3f}]")
    print(f"--- Stage {st}: paired differences (Hybrid minus X) ---")
    r = res[(res.stage == st) & (res.comparison == "difference")]
    for _, x in r.iterrows(): print(f"{x.system:20s} {x.metric:14s} {x.estimate:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}]")

man = json.load(open(os.path.join(D, "subjects_manifest.json")))
ch = pd.Series({s: v["channels"] for s, v in man.items()})
dfb = pd.read_csv(os.path.join(D, "stageB_long.csv")); PB = participant_table(dfb, MODELS)
rows = []
for (m, k) in PB.columns:
    if k == "residual_pcc": continue
    v = PB[(m, k)]; rho, p = spearmanr(ch.loc[v.index], v.values)
    rows.append(dict(system=m, metric=k, spearman_rho=rho, p_value=p, n=len(v)))
cs = pd.DataFrame(rows); cs.to_csv(os.path.join(D, "supp_channels_spearman.csv"), index=False)
print("\n=== Stage B: Spearman correlation between number of channels and participant mean (n = 10, uncorrected) ===")
print(cs.round(3).to_string(index=False))
print("\nchannels:", ch.to_dict())
