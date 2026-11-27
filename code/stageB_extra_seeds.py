"""
stageB_extra_seeds.py -- STAGE B, ADDITIONAL SEEDS (supplementary material).

The main Stage B runs (stageB_subject.py) use seed 42. This script repeats exactly the same participant-folds with other seeds
(default 123 and 456, the seeds of Stage A). Nothing else changes: it imports stageB_subject.py (not modified), so the split, windows,
architectures, loss, optimiser, scheduler, early stopping, metrics and the template are identical, and the learning rate is the one
fixed a priori in stageB_subject.py (FIXED_LR).
The seed-42 results in <SEEG_OUT_DIR>/stageB are never touched: new results go to <SEEG_OUT_DIR>/stageB_seeds/<model>_<sub>_fold<k>_seed<s>.json
(raw predictions are not saved here). The linear model is deterministic and is not repeated.

Usage (SEEG_OUT_DIR set; all scripts in the same folder):
    python stageB_extra_seeds.py smoke run hybrid                 # 1 epoch, 2 participants x 2 folds: plumbing check
    python stageB_extra_seeds.py run hybrid                       # seeds 123 and 456, all 10 participants x 10 folds
    python stageB_extra_seeds.py run hybrid 123                   # only seed 123
    python stageB_extra_seeds.py run hybrid 123 sub-03 sub-04     # seed 123, only these participants
    python stageB_extra_seeds.py summary                          # table over seeds 42 / 123 / 456 -> stageB_seeds_summary.csv
Runs are resumable (finished participant-folds are skipped). Order suggested: hybrid, baseline, neuroincept, convedsr.
Optional: STAGEB_THREADS=4 limits the CPU threads.
"""
import os, sys, json, time, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageB_subject as B                      # also imports stageA_pooled / stageA_extra (registers all models)
A = B.A

SEEDS_DEFAULT = (123, 456)
RES_S = os.path.join(B.OUT, "stageB_seeds_smoke" if B.SMOKE else "stageB_seeds")
os.makedirs(RES_S, exist_ok=True)


def run(model, seeds, subs, folds, log):
    lr = B.fixed_lr(model)
    log(f"=== stage B extra seeds | {model} | lr {lr} | seeds {seeds} | participants {subs} | folds {folds}")
    for seed in seeds:
        for sub in subs:
            for k in folds:
                tag = f"{model}_{sub}_fold{k}_seed{seed}"
                if os.path.exists(os.path.join(RES_S, tag + ".json")):
                    continue
                t0 = time.time(); B.set_fold(k)
                D, ymu, ysd = A.load_all([sub])
                m, v = A.train_model(model, lr, seed, D, 1 if B.SMOKE else A.MAX_EP,
                                     lambda s: log(f"  {sub} fold {k} seed {seed}: " + s.strip()))
                wins = A.windows(D, "test")
                P_raw = A.predict(m, D, wins) * ysd[None, :, None] + ymu[None, :, None]
                res, _ = A.evaluate(D, wins, P_raw)
                r = res[sub]
                json.dump(dict(model=model, lr=lr, seed=seed, subject=sub, fold=k, per_subject={sub: r}),
                          open(os.path.join(RES_S, tag + ".json"), "w"), indent=1)
                log(f"{tag}: PCC {r['pcc']:.3f} | template {r['template_pcc']:.3f} | in speech {r['pcc_in_speech']:.3f} "
                    f"(template {r['template_in_speech']:.3f}) | residual {r['residual_pcc']:+.3f} | {time.time() - t0:.0f}s")


def load_all_seeds():
    """seed 42 from stageB/, other seeds from stageB_seeds/ ; one row per (model, seed, participant, fold)"""
    import pandas as pd
    df42 = B.load_runs_b(smoke=B.SMOKE)
    df42 = df42[df42.model != "linear"]
    rows = []
    for p in sorted(glob.glob(os.path.join(RES_S, "*.json"))):
        j = json.load(open(p))
        for sub, r in j["per_subject"].items():
            rows.append(dict(model=j["model"], lr=j["lr"], seed=j["seed"], fold=j["fold"], subject=sub, **r))
    return pd.concat([df42, pd.DataFrame(rows)], ignore_index=True)


def summary():
    import pandas as pd
    df = load_all_seeds()
    if not len(df):
        print("no finished folds yet"); return
    keys = ["pcc", "pcc_in_speech", "residual_pcc"]
    cnt = df.groupby(["model", "seed"])["fold"].count().unstack()
    print("finished participant-folds per model and seed (100 = complete):\n", cnt.to_string(), "\n")
    # participant value = mean over its folds; model/seed value = mean over participants; only complete (model, seed) cells
    part = df.groupby(["model", "seed", "subject"])[keys].mean().reset_index()
    full = cnt[cnt >= 100].stack().index
    ms = part.groupby(["model", "seed"])[keys].mean().reset_index()
    ms = ms[[(a, b) in set(full) for a, b in zip(ms.model, ms.seed)]]
    print("mean over the 10 participants, per seed (complete runs only):\n", ms.round(3).to_string(index=False), "\n")
    agg = ms.groupby("model")[keys].agg(["mean", "std", "count"]).round(3)
    print("across seeds (std = sample SD over seeds):\n", agg.to_string(), "\n")
    out = os.path.join(B.OUT, "stageB_seeds_summary.csv")
    ms.to_csv(out, index=False); print("written:", out)
    # per-seed Hybrid minus Baseline (paired by participant) when both are complete for the seed
    for s in sorted(ms.seed.unique()):
        h = part[(part.model == "hybrid") & (part.seed == s)].set_index("subject")
        b = part[(part.model == "baseline") & (part.seed == s)].set_index("subject")
        if len(h) == 10 and len(b) == 10:
            d = (h[keys] - b[keys])
            print(f"seed {s}: Hybrid - Baseline  PCC {d.pcc.mean():+.3f} ({(d.pcc > 0).sum()}/10 participants higher) | "
                  f"in-speech {d.pcc_in_speech.mean():+.3f} ({(d.pcc_in_speech > 0).sum()}/10)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    if args[:1] == ["summary"]:
        summary(); sys.exit()
    assert args[:1] == ["run"] and len(args) > 1 and args[1] in B.MODELS and args[1] != "linear", __doc__
    model, rest = args[1], args[2:]
    seeds = [int(rest.pop(0))] if rest and rest[0].isdigit() else list(SEEDS_DEFAULT)
    subs = rest or (["sub-02", "sub-06"] if B.SMOKE else A.ALL_SUBS)
    folds = [0, 1] if B.SMOKE else list(range(A.NBLK))
    logf = open(os.path.join(RES_S, f"log_{model}.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    run(model, seeds, subs, folds, log)
    summary()
