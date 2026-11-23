"""
stageB_purge.py -- STAGE B, BLOCK-BOUNDARY (LEAKAGE) CONTROL (supplementary material).

Question: does anything leak between the training blocks and the held-out blocks across the block boundaries (the signal and the task are
continuous in time, and the neighbouring blocks are only separated by the block edge)?  Test: retrain some participant-folds after REMOVING
from the training set every training window that lies closer than GAP frames to the TEST block or to the VALIDATION block of that fold
(GAP = 86 frames ~ 1 s, 256 frames ~ 3 s = one trial period). Test and validation windows, the learning rate, the seed (42), the optimiser, the metrics and
the template are exactly those of the main Stage B run (stageB_subject.py, not modified), so the result is directly comparable, fold by
fold, with the main result in <SEEG_OUT_DIR>/stageB.  GAP = 0 re-runs the main protocol (sanity check of this script: it should reproduce the main result).
Note: train windows start every 64 frames inside their block, so the real distance between a kept window and the held-out block is >= GAP
(it is logged: 'min distance').  Linear model: see stageB_verwoert_linear.py ('real_purged').
Outputs: <SEEG_OUT_DIR>/stageB_purge/<model>_<sub>_fold<k>_gap<G>.json (raw predictions are not saved).

Usage (SEEG_OUT_DIR set; all scripts in the same folder):
    python stageB_purge.py smoke run hybrid 86                                  # 1 epoch, 2 participants x 2 folds: plumbing check
    python stageB_purge.py run hybrid 86 256                                    # default: sub-02 sub-06 sub-10, all 10 folds
    python stageB_purge.py run hybrid 86 --subs sub-02 sub-06 --folds 2 3 4     # a few folds only
    python stageB_purge.py run hybrid 0 --subs sub-02 --folds 3                 # sanity: GAP 0 must reproduce the main run
    python stageB_purge.py summary                                              # paired comparison with the main run -> stageB_purge_summary.csv
Runs are resumable.  Use STAGEB_LR_NEUROINCEPT=1e-4 for neuroincept, as in the main Stage B run.  STAGEB_THREADS=4 limits CPU threads.
"""
import os, sys, json, time, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageB_subject as B
A = B.A
ORIG_WINDOWS = A.windows

RES_P = os.path.join(B.OUT, "stageB_purge_smoke" if B.SMOKE else "stageB_purge")
os.makedirs(RES_P, exist_ok=True)
DEFAULT_SUBS = ["sub-02", "sub-06", "sub-10"]


def purged_windows(gap):
    """a replacement for A.windows: test / validation windows unchanged; training windows closer than `gap` frames to the test or validation block are dropped"""
    def w(D, kind):
        out = ORIG_WINDOWS(D, kind)
        if kind != "train" or gap <= 0:
            return out
        keep = []
        for sub, st in out:
            blocks = D[sub]["blocks"]
            if all(not (st < blocks[b][1] + gap and st + A.W > blocks[b][0] - gap) for b in A.TEST_B + A.VAL_B):
                keep.append((sub, st))
        return keep
    return w


def min_distance(D, wins):
    """smallest distance (frames) between any training window and the test / validation block"""
    best = 10 ** 9
    for sub, st in wins:
        for b in A.TEST_B + A.VAL_B:
            s, e = D[sub]["blocks"][b]
            best = min(best, max(s - (st + A.W), st - e))
    return best


def run(model, gaps, subs, folds, log):
    lr = B.stage_a_lr(model)
    log(f"=== stage B purge | {model} | lr {lr} | gaps {gaps} | participants {subs} | folds {folds}")
    for gap in gaps:
        for sub in subs:
            for k in folds:
                tag = f"{model}_{sub}_fold{k}_gap{gap}"
                if os.path.exists(os.path.join(RES_P, tag + ".json")):
                    continue
                t0 = time.time(); B.set_fold(k)
                A.windows = ORIG_WINDOWS
                D, ymu, ysd = A.load_all([sub])
                n0 = len(ORIG_WINDOWS(D, "train"))
                A.windows = purged_windows(gap)                      # train_model() looks A.windows up at call time
                n1 = len(A.windows(D, "train")); md = min_distance(D, A.windows(D, "train"))
                log(f"  {tag}: train windows {n0} -> {n1} (removed {n0 - n1}); min distance to held-out blocks {md} frames ({md / 86.13:.2f} s)")
                m, v = A.train_model(model, lr, B.SEED, D, 1 if B.SMOKE else A.MAX_EP, lambda s: log(f"  {sub} fold {k} gap {gap}: " + s.strip()))
                A.windows = ORIG_WINDOWS
                wins = A.windows(D, "test")
                P_raw = A.predict(m, D, wins) * ysd[None, :, None] + ymu[None, :, None]
                res, _ = A.evaluate(D, wins, P_raw)
                r = res[sub]
                json.dump(dict(model=model, lr=lr, seed=B.SEED, subject=sub, fold=k, gap=gap, n_train_windows=n1, n_removed=n0 - n1,
                               min_distance=int(md), per_subject={sub: r}), open(os.path.join(RES_P, tag + ".json"), "w"), indent=1)
                log(f"{tag}: PCC {r['pcc']:.3f} | template {r['template_pcc']:.3f} | in speech {r['pcc_in_speech']:.3f} "
                    f"(template {r['template_in_speech']:.3f}) | residual {r['residual_pcc']:+.3f} | {time.time() - t0:.0f}s")


def summary():
    import pandas as pd
    main = B.load_runs_b(smoke=B.SMOKE)
    rows = []
    for p in sorted(glob.glob(os.path.join(RES_P, "*.json"))):
        j = json.load(open(p))
        for sub, r in j["per_subject"].items():
            rows.append(dict(model=j["model"], gap=j["gap"], fold=j["fold"], subject=sub, n_removed=j["n_removed"], min_distance=j["min_distance"], **r))
    if not rows:
        print("no finished folds yet"); return
    pg = pd.DataFrame(rows)
    keys = ["pcc", "pcc_in_speech", "residual_pcc"]
    m = main[["model", "fold", "subject"] + keys].rename(columns={k: k + "_main" for k in keys})
    df = pg.merge(m, on=["model", "fold", "subject"], how="inner")
    print("paired with the main Stage B run (same model / participant / fold, seed 42):")
    out = []
    for (mod, g), d in df.groupby(["model", "gap"]):
        row = dict(model=mod, gap=g, gap_s=round(g / 86.13, 2), n_pairs=len(d), mean_removed_windows=round(d.n_removed.mean(), 1))
        for k in keys:
            diff = d[k] - d[k + "_main"]
            row.update({k + "_main": round(d[k + "_main"].mean(), 3), k + "_purged": round(d[k].mean(), 3),
                        k + "_diff_mean": round(diff.mean(), 3), k + "_diff_sd": round(diff.std(ddof=1), 3) if len(d) > 1 else float("nan"),
                        k + "_diff_min": round(diff.min(), 3), k + "_diff_max": round(diff.max(), 3)})
        out.append(row)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(res.to_string(index=False))
    f = os.path.join(B.OUT, "stageB_purge_summary.csv"); res.to_csv(f, index=False); print("\nwritten:", f)
    print("\nper participant-fold (PCC purged vs main):")
    print(df[["model", "gap", "subject", "fold", "pcc_main", "pcc", "pcc_in_speech_main", "pcc_in_speech"]].round(3).to_string(index=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    if args[:1] == ["summary"]:
        summary(); sys.exit()
    assert args[:1] == ["run"] and len(args) > 2 and args[1] in B.MODELS and args[1] != "linear", __doc__
    model = args[1]; gaps, subs, folds, mode = [], [], [], "gaps"
    for a in args[2:]:
        if a == "--subs": mode = "subs"; continue
        if a == "--folds": mode = "folds"; continue
        {"gaps": gaps, "subs": subs, "folds": folds}[mode].append(int(a) if mode != "subs" else a)
    assert gaps, "give at least one GAP (frames), e.g. 86 256"
    subs = subs or (["sub-02", "sub-06"] if B.SMOKE else DEFAULT_SUBS)
    folds = folds or ([0, 1] if B.SMOKE else list(range(A.NBLK)))
    logf = open(os.path.join(RES_P, f"log_{model}.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    run(model, gaps, subs, folds, log)
    summary()
