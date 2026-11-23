"""
stageB_lr.py -- STAGE B, LEARNING-RATE SENSITIVITY (supplementary material).

Main Stage B uses, for every model, the learning rate chosen in Stage A (validation MSE of the pooled model), except NeuroIncept (1e-4, a declared
override). Two questions: (1) the Stage-A validation blocks (blocks 1 and 6 in the paper, 0 and 5 in the code) are the test blocks of Stage-B folds 1
and 6, so the choice among the three grid values has seen those blocks as validation data; does the result of those folds depend on that choice?
(2) is the NeuroIncept result (low PCC, collapse for some participants) due to the declared override 1e-4?  This script re-runs participant-folds
with an explicitly given learning rate (seed 42 unless told otherwise); everything else is the main Stage B protocol (stageB_subject.py, not
modified), so results are directly comparable, fold by fold, with <SEEG_OUT_DIR>/stageB.
Outputs: <SEEG_OUT_DIR>/stageB_lr/<model>_<sub>_fold<k>_lr<lr>_seed<s>.json (raw predictions are not saved).

Usage (SEEG_OUT_DIR set; all scripts in the same folder). Folds are 0-indexed as in the code (paper fold = code fold + 1):
    python stageB_lr.py smoke run neuroincept 1e-3                                  # plumbing check (1 epoch)
    python stageB_lr.py run neuroincept 1e-3                                        # all 10 participants x 10 folds
    python stageB_lr.py run hybrid 3e-4 1e-3 --folds 0 5                            # the two folds whose test block was a Stage-A validation block
    python stageB_lr.py run baseline 3e-4 1e-3 --folds 0 5 --subs sub-02 sub-06
    python stageB_lr.py summary
Runs are resumable.  STAGEB_THREADS=4 limits CPU threads.
"""
import os, sys, json, time, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageB_subject as B
A = B.A

RES_L = os.path.join(B.OUT, "stageB_lr_smoke" if B.SMOKE else "stageB_lr")
os.makedirs(RES_L, exist_ok=True)


def run(model, lrs, subs, folds, seed, log):
    log(f"=== stage B lr sensitivity | {model} | lrs {lrs} | seed {seed} | participants {subs} | folds {folds}")
    for lr in lrs:
        for sub in subs:
            for k in folds:
                tag = f"{model}_{sub}_fold{k}_lr{lr:g}_seed{seed}"
                if os.path.exists(os.path.join(RES_L, tag + ".json")):
                    continue
                t0 = time.time(); B.set_fold(k)
                D, ymu, ysd = A.load_all([sub])
                m, v = A.train_model(model, lr, seed, D, 1 if B.SMOKE else A.MAX_EP, lambda s: log(f"  {sub} fold {k} lr {lr:g}: " + s.strip()))
                wins = A.windows(D, "test")
                P_raw = A.predict(m, D, wins) * ysd[None, :, None] + ymu[None, :, None]
                res, _ = A.evaluate(D, wins, P_raw)
                r = res[sub]
                json.dump(dict(model=model, lr=lr, seed=seed, subject=sub, fold=k, per_subject={sub: r}), open(os.path.join(RES_L, tag + ".json"), "w"), indent=1)
                log(f"{tag}: PCC {r['pcc']:.3f} | template {r['template_pcc']:.3f} | in speech {r['pcc_in_speech']:.3f} "
                    f"(template {r['template_in_speech']:.3f}) | residual {r['residual_pcc']:+.3f} | {time.time() - t0:.0f}s")


def summary():
    import pandas as pd
    main = B.load_runs_b(smoke=B.SMOKE)
    rows = []
    for p in sorted(glob.glob(os.path.join(RES_L, "*.json"))):
        j = json.load(open(p))
        for sub, r in j["per_subject"].items():
            rows.append(dict(model=j["model"], lr_new=j["lr"], seed=j["seed"], fold=j["fold"], subject=sub, **r))
    if not rows:
        print("no finished folds yet"); return
    new = pd.DataFrame(rows)
    keys = ["pcc", "pcc_in_speech", "residual_pcc"]
    m = main[main.seed == 42][["model", "fold", "subject", "lr"] + keys].rename(columns={"lr": "lr_main", **{k: k + "_main" for k in keys}})
    df = new.merge(m, on=["model", "fold", "subject"], how="inner")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    out = []
    for (mod, lr), d in df.groupby(["model", "lr_new"]):
        row = dict(model=mod, lr_main=d.lr_main.iloc[0], lr_new=lr, n_pairs=len(d))
        for k in keys:
            row.update({k + "_main": round(d[k + "_main"].mean(), 3), k + "_new": round(d[k].mean(), 3), k + "_diff": round((d[k] - d[k + "_main"]).mean(), 3)})
        row["folds_below_0.3_main"] = int((d.pcc_main < 0.3).sum()); row["folds_below_0.3_new"] = int((d.pcc < 0.3).sum())
        out.append(row)
    res = pd.DataFrame(out)
    print("same participant-folds, main lr vs new lr (seed 42):\n", res.to_string(index=False))
    f = os.path.join(B.OUT, "stageB_lr_summary.csv"); res.to_csv(f, index=False); print("\nwritten:", f)
    print("\nper participant (mean over the finished folds of that participant):")
    pp = df.groupby(["model", "lr_new", "subject"])[["pcc_main", "pcc", "pcc_in_speech_main", "pcc_in_speech"]].mean().round(3)
    print(pp.to_string())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    if args[:1] == ["summary"]:
        summary(); sys.exit()
    assert args[:1] == ["run"] and len(args) > 2 and args[1] in B.MODELS and args[1] != "linear", __doc__
    model = args[1]; lrs, subs, folds, seeds, mode = [], [], [], [], "lrs"
    for a in args[2:]:
        if a in ("--subs", "--folds", "--seed"): mode = a[2:] + "s" if a != "--seed" else "seeds"; continue
        {"lrs": lrs, "subss": subs, "foldss": folds, "seeds": seeds}[mode].append(float(a) if mode == "lrs" else (a if mode == "subss" else int(a)))
    assert lrs, "give at least one learning rate, e.g. 1e-3"
    subs = subs or (["sub-02", "sub-06"] if B.SMOKE else A.ALL_SUBS)
    folds = folds or ([0, 1] if B.SMOKE else list(range(A.NBLK)))
    seed = seeds[0] if seeds else B.SEED
    logf = open(os.path.join(RES_L, f"log_{model}.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    run(model, lrs, subs, folds, seed, log)
    summary()
