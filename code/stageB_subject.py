"""
stageB_subject.py -- STAGE B: SUBJECT-DEPENDENT models, 10 contiguous folds per participant.

One model per participant and per fold (nothing is shared between participants). Models: linear (Verwoert-style), hybrid (MR-ResBlocks),
baseline (CNN-BiLSTM), neuroincept, convedsr. It re-uses the Stage-A harness (stageA_pooled.py / stageA_extra.py, not modified), so the
architectures, loss, optimiser, weight decay, scheduler, early stopping, windows, metrics and the template reference are the same.

Protocol, per participant (10 folds, k = 0..9)
  * the recording is cut into the same 10 contiguous blocks as in Stage A.
  * fold k: TEST = block k, VALIDATION = block (k+1) mod 10 (used only for early stopping / lr scheduler), TRAIN = the other 8 blocks.
    Windows never cross block boundaries. Windows: 256 frames, train stride 64, val/test stride 256 (as Stage A).
  * every participant-fold is trained from scratch (seed 42). The learning rate is FIXED A PRIORI (no selection on any data block):
    CNN-BiLSTM and MR-CNN-BiLSTM 1e-4 (our earlier setup), NeuroIncept 1e-4 and ConvED-SR 5e-4 (the rates of their original code).
  * inputs are z-scored with TRAIN-block statistics of that participant; mel targets are standardised with the train blocks of that
    participant; the participant's own channels are zero-padded to 127 with a mask (the padding is inert here: all channels of one
    participant are always real).
  * metrics per participant-fold on the test frames: PCC per mel bin over the concatenated test frames, mean over the 80 bins
    (the Verwoert / ConvED-SR definition); the same inside speech; the time-in-trial template and the residual PCC. The
    participant's value = mean over its 10 folds. Raw predictions are saved (log-mel) for the STOI step.
  * the Verwoert-style linear model: the same stacking (+-4, step 5 -> 9 x channels, edges of the recording replicated), PCA 50, linear
    regression, fitted on all frames of the 8 train blocks, evaluated on exactly the same test frames as the neural models.
    As in Verwoert's code, the stacked inputs of train frames next to the test block may overlap with test inputs by up to 20 frames
    (the neural windows do not).
  * test frames: those covered by 256-frame test windows (at most 255 frames at the end of each block are not scored, for every model).

Usage (SEEG_OUT_DIR set; stageA_pooled.py, stageA_extra.py, template_floor_continuous.py, stageB_verwoert_linear.py in the same folder):
    python stageB_subject.py smoke run hybrid          # 1 epoch, 2 participants x 2 folds: checks the plumbing and the time per epoch
    python stageB_subject.py run linear                # all 10 participants x 10 folds
    python stageB_subject.py run hybrid                # (also: baseline, neuroincept, convedsr)  -- resumable: finished folds are skipped
    python stageB_subject.py run hybrid sub-03 sub-04  # only these participants
    python stageB_subject.py summary
Optional: STAGEB_THREADS=4 limits the CPU threads (useful while another job is running).
"""
import os, sys, json, time, glob, re
import numpy as np

if os.environ.get("STAGEB_THREADS"):
    import torch as _t
    _t.set_num_threads(int(os.environ["STAGEB_THREADS"]))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageA_pooled as A
import stageA_extra as E                                  # registers neuroincept / convedsr and patches A.train_model

OUT = A.OUT
SMOKE = A.SMOKE
RES = os.path.join(OUT, "stageB_smoke" if SMOKE else "stageB")
os.makedirs(RES, exist_ok=True)
SEED = 42
MODELS = ("linear", "hybrid", "baseline", "neuroincept", "convedsr")


# Learning rates fixed a priori (no selection on validation or test data):
#   baseline (CNN-BiLSTM) and hybrid (MR-CNN-BiLSTM): 1e-4, as in our earlier setup;
#   neuroincept: 1e-4 and convedsr: 5e-4, the rates of their original public code.
FIXED_LR = {"baseline": 1e-4, "hybrid": 1e-4, "neuroincept": 1e-4, "convedsr": 5e-4, "convedsr1024": 5e-4}


def fixed_lr(model):
    """the a-priori learning rate of this model (no data are used to choose it)"""
    if SMOKE:
        return 3e-4
    key = "convedsr" if model.startswith("convedsr") else model
    return FIXED_LR[key]


stage_a_lr = fixed_lr      # old name, kept so that older notebooks still run


def set_fold(k):
    A.TEST_B, A.VAL_B = (k,), ((k + 1) % A.NBLK,)


def run_linear_fold(D, sub, ymu, ysd):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    d = D[sub]; X = d["X"]; T = len(X)
    offs = np.arange(-4, 5) * 5
    idx = np.clip(np.arange(T)[:, None] + offs[None, :], 0, T - 1)
    tr = d["train_idx"]
    te_wins = A.windows(D, "test")
    te = np.concatenate([np.arange(st, st + A.W) for _, st in te_wins])

    def stack(rows):
        return X[idx[rows]].reshape(len(rows), -1)

    Xtr = stack(tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    pca = PCA(50, random_state=0).fit((Xtr - mu) / sd)
    lr = LinearRegression().fit(pca.transform((Xtr - mu) / sd), d["Yn"][tr])
    Pn = lr.predict(pca.transform((stack(te) - mu) / sd))
    P_raw = (Pn * ysd + ymu).reshape(len(te_wins), A.W, A.MEL).transpose(0, 2, 1)
    return te_wins, P_raw


def run(model, subs, folds, log):
    lr = fixed_lr(model) if model != "linear" else None
    log(f"=== stage B | {model} | lr {lr} | participants {subs} | folds {folds}")
    for sub in subs:
        for k in folds:
            tag = f"{model}_{sub}_fold{k}"
            if os.path.exists(os.path.join(RES, tag + ".json")):
                continue
            t0 = time.time(); set_fold(k)
            D, ymu, ysd = A.load_all([sub])
            if model == "linear":
                wins, P_raw = run_linear_fold(D, sub, ymu, ysd)
            else:
                m, v = A.train_model(model, lr, SEED, D, 1 if SMOKE else A.MAX_EP, lambda s: log(f"  {sub} fold {k}: " + s.strip()))
                wins = A.windows(D, "test")
                P_raw = A.predict(m, D, wins) * ysd[None, :, None] + ymu[None, :, None]
            res, preds = A.evaluate(D, wins, P_raw)
            r = res[sub]
            json.dump(dict(model=model, lr=lr, seed=SEED, subject=sub, fold=k, per_subject={sub: r}), open(os.path.join(RES, tag + ".json"), "w"), indent=1)
            np.savez_compressed(os.path.join(RES, tag + "_preds.npz"), **preds)
            log(f"{tag}: PCC {r['pcc']:.3f} | template {r['template_pcc']:.3f} | in speech {r['pcc_in_speech']:.3f} (template {r['template_in_speech']:.3f}) "
                f"| residual {r['residual_pcc']:+.3f} | {time.time() - t0:.0f}s")


def load_runs_b(out=None, smoke=False):
    """DataFrame, one row per (model, participant, fold); same columns as stageA_report.load_runs (+ 'fold')"""
    import pandas as pd
    d = os.path.join(out or OUT, "stageB_smoke" if smoke else "stageB")
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(p))
        for sub, r in j["per_subject"].items():
            rows.append(dict(model=j["model"], lr=j.get("lr") or float("nan"), seed=j["seed"], fold=j["fold"], subject=sub, **r))
    df = pd.DataFrame(rows)
    if len(df):
        df["pcc_minus_template"] = df["pcc"] - df["template_pcc"]
        df["speech_minus_template"] = df["pcc_in_speech"] - df["template_in_speech"]
    return df


def summary():
    df = load_runs_b(smoke=SMOKE)
    if not len(df):
        print("no finished folds yet"); return
    cnt = df.groupby("model")["fold"].count()
    print("finished participant-folds per model (100 = complete):\n", cnt.to_string())
    keys = ["pcc", "template_pcc", "residual_pcc", "pcc_in_speech", "template_in_speech"]
    print(df.groupby("subject")[keys].mean().round(3).to_string() if df.model.nunique() == 1 else
          df.groupby("model")[keys].mean().round(3).to_string())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    if args[:1] == ["summary"]:
        summary(); sys.exit()
    assert args[:1] == ["run"] and args[1] in MODELS, __doc__
    model = args[1]
    subs = args[2:] or (["sub-02", "sub-06"] if SMOKE else A.ALL_SUBS)
    folds = [0, 1] if SMOKE else list(range(A.NBLK))
    logf = open(os.path.join(RES, f"log_{model}.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    run(model, subs, folds, log)
    summary()
