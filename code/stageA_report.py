"""
stageA_report.py -- read-only helpers that turn the Stage-A result files into readable tables and figures.
Used by StageA_results.ipynb. It never trains anything and never changes the result files.

Per-model rows are labelled with the model's real name; per-participant tables show every model next to the no-neural-input
"template" reference. All numbers are PCC per mel bin (averaged over the 80 bins) on the held-out test blocks.
"""
import os, re, glob, json
import numpy as np
import pandas as pd

LABEL = {"template": "Template (no neural input)", "linear": "Linear (Verwoert-style, pooled)", "baseline": "Baseline CNN-BiLSTM",
         "hybrid": "Hybrid (MR-ResBlocks)", "neuroincept": "NeuroIncept (re-impl.)", "convedsr": "ConvED-SR-style (re-impl.)",
         "baseline_null": "Baseline, noise input", "hybrid_null": "Hybrid, noise input",
         "neuroincept_null": "NeuroIncept, noise input", "convedsr_null": "ConvED-SR-style, noise input"}
ORDER = list(LABEL)
METRICS = ["pcc", "template_pcc", "pcc_in_speech", "template_in_speech", "residual_pcc", "mse"]


def res_dir(out=None, smoke=False):
    out = out or os.environ["SEEG_OUT_DIR"]
    return os.path.join(out, "stageA_smoke" if smoke else "stageA")


def load_runs(out=None, smoke=False):
    rows = []
    for p in sorted(glob.glob(os.path.join(res_dir(out, smoke), "*.json"))):
        j = json.load(open(p))
        for sub, r in j["per_subject"].items():
            rows.append(dict(model=j["model"], lr=j.get("lr", np.nan), seed=j.get("seed", -1), subject=sub, **r))
    df = pd.DataFrame(rows)
    if len(df):
        df["pcc_minus_template"] = df["pcc"] - df["template_pcc"]
        df["speech_minus_template"] = df["pcc_in_speech"] - df["template_in_speech"]
    return df


def status(out=None, smoke=False):
    """which runs are finished + tail of the logs"""
    d = res_dir(out, smoke)
    done = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(d, "*.json")))
    print(f"finished runs ({len(done)}):"); [print("  ", n) for n in done]
    for lg in sorted(glob.glob(os.path.join(d, "log_*.txt"))):
        lines = open(lg, errors="replace").read().splitlines()
        print(f"\n--- {os.path.basename(lg)} (last 4 lines)"); print("\n".join(lines[-4:]))


def model_table(df):
    """one row per model: mean over participants, then mean +- sd over seeds"""
    out = []
    for m in [x for x in ORDER if x != "template" and x in set(df["model"])]:
        d = df[df["model"] == m]
        per_seed = d.groupby("seed")[METRICS + ["pcc_minus_template", "speech_minus_template"]].mean()
        row = dict(model=LABEL[m], seeds=len(per_seed), lr=d["lr"].iloc[0])
        for k in ["pcc", "template_pcc", "pcc_minus_template", "pcc_in_speech", "template_in_speech", "speech_minus_template", "residual_pcc"]:
            row[k] = f"{per_seed[k].mean():+.3f}" + (f" ± {per_seed[k].std(ddof=0):.3f}" if len(per_seed) > 1 else "")
        out.append(row)
    t = pd.DataFrame(out).set_index("model")
    return t.rename(columns={"pcc": "PCC", "template_pcc": "template PCC", "pcc_minus_template": "PCC − template",
                             "pcc_in_speech": "PCC in speech", "template_in_speech": "template in speech",
                             "speech_minus_template": "in speech − template", "residual_pcc": "residual PCC"})


def subject_table(df, metric="pcc", models=None):
    """rows = participants, columns = models (mean over seeds) + the template reference"""
    models = models or [m for m in ORDER if m not in ("template",) and m in set(df["model"])]
    cols = {}
    for m in models:
        cols[LABEL[m]] = df[df["model"] == m].groupby("subject")[metric].mean()
    ref = {"pcc": "template_pcc", "pcc_in_speech": "template_in_speech"}.get(metric)
    if ref:
        cols[LABEL["template"]] = df.groupby("subject")[ref].mean()
    t = pd.DataFrame(cols).round(3)
    t.loc["mean"] = t.mean().round(3)
    return t


def paired(df, a, b, metric="pcc"):
    """participant-level paired comparison of two models (seed-averaged); exact two-sided Wilcoxon signed-rank"""
    from scipy.stats import wilcoxon
    x = df[df["model"] == a].groupby("subject")[metric].mean(); y = df[df["model"] == b].groupby("subject")[metric].mean()
    s = x.index.intersection(y.index); d = (x[s] - y[s])
    out = dict(comparison=f"{LABEL[a]}  minus  {LABEL[b]}", metric=metric, n=len(s), mean_diff=round(d.mean(), 4),
               a_better_in=f"{int((d > 0).sum())}/{len(s)}")
    try:
        out["p_exact_two_sided"] = round(wilcoxon(d, method="exact").pvalue, 4) if len(s) >= 5 and (d != 0).any() else float("nan")
    except Exception as e:
        out["p_exact_two_sided"] = float("nan")
    return out, d.round(4)


def paired_table(df, pairs, metrics=("pcc", "pcc_in_speech", "residual_pcc")):
    rows = []
    for a, b in pairs:
        if a in set(df["model"]) and b in set(df["model"]):
            for m in metrics:
                rows.append(paired(df, a, b, m)[0])
    return pd.DataFrame(rows)


def plot_subjects(df, metric="pcc", models=("baseline", "hybrid", "linear", "neuroincept", "convedsr"), title=None):
    import matplotlib.pyplot as plt
    t = subject_table(df, metric, [m for m in models if m in set(df["model"])]).drop(index="mean")
    ax = t.plot.bar(figsize=(12, 4.5), width=0.85)
    ax.set_ylabel(metric.replace("_", " ")); ax.set_title(title or f"{metric} per participant (mean over seeds)")
    ax.legend(fontsize=8, ncol=2); plt.xticks(rotation=0); plt.tight_layout(); plt.show()


def learning_curves(out=None, smoke=False):
    """validation MSE per epoch for every training run found in the logs"""
    import matplotlib.pyplot as plt
    pat = re.compile(r"\[(\w+) lr=([0-9.e+-]+) seed=(\d+)\] epoch (\d+)/\d+: .*val MSE ([0-9.]+)")
    runs = {}
    for lg in glob.glob(os.path.join(res_dir(out, smoke), "log_*.txt")):
        for line in open(lg, errors="replace"):
            m = pat.search(line)
            if m:
                runs.setdefault((m.group(1), m.group(2), m.group(3)), []).append((int(m.group(4)), float(m.group(5))))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for (name, lr, seed), v in sorted(runs.items()):
        v = sorted(set(v)); ax.plot([a for a, _ in v], [b for _, b in v], label=f"{name} lr={lr} s={seed}", lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation MSE (standardised mel)"); ax.legend(fontsize=6, ncol=3); plt.tight_layout(); plt.show()
    return len(runs)


def per_bin_pcc(out=None, model="hybrid", smoke=False):
    """per-mel-bin PCC from the saved predictions (mean over participants), model vs template"""
    import matplotlib.pyplot as plt
    d = res_dir(out, smoke); files = sorted(glob.glob(os.path.join(d, f"{model}_lr*_seed*_preds.npz")))
    if not files:
        print("no saved predictions for", model); return
    z = np.load(files[0]); subs = sorted({k[:6] for k in z.files if k.endswith("_idx")}); P = []; T = []
    spec = {s: np.load(os.path.join(out or os.environ["SEEG_OUT_DIR"], f"{s}_spec.npy")) for s in subs}
    pc = lambda a, b: ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / (np.sqrt(((a - a.mean(0)) ** 2).sum(0) * ((b - b.mean(0)) ** 2).sum(0)) + 1e-12)
    for s in subs:
        Y = spec[s][z[s + "_idx"]]; P.append(pc(Y, z[s + "_P"])); T.append(pc(Y, z[s + "_T"]))
    fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(np.mean(P, 0), label=LABEL[model]); ax.plot(np.mean(T, 0), label=LABEL["template"])
    ax.set_xlabel("mel bin (0 = lowest frequency)"); ax.set_ylabel("PCC"); ax.legend(); plt.tight_layout(); plt.show()
