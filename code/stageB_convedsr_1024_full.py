"""
stageB_convedsr_1024_full.py -- ConvED-SR with its ORIGINAL input resolution (the HG envelope at 1024 Hz), for ALL participants,
all 10 folds and the three seeds (42, 123, 456). This becomes the main ConvED-SR of the paper.

Identical to stageB_convedsr_1024.py (same network, same 1024 Hz HG envelope, last encoder kernel (2,2), same folds, windows, loss,
optimiser, scheduler, early stopping, metrics and template; learning rate 5e-4 from FIXED_LR), but:
  * runs every participant x fold x seed, and SAVES the test predictions (needed by inword_stageB.py and quality_metrics.py);
  * results go to <SEEG_OUT_DIR>/stageB_1024full/convedsr_<sub>_fold<k>_seed<s>.json (+ _preds.npz); runs are resumable;
  * "install" copies them to the places the analysis scripts read (stageB/ for seed 42, stageB_seeds_preds/ for 123 and 456),
    after moving the HG-frame ConvED-SR files of those folders to archive_convedsr_hgframes/ (nothing is deleted).

Usage (SEEG_OUT_DIR and SEEG_DATA_ROOT set; all scripts in the same folder):
    python stageB_convedsr_1024_full.py run                 # all participants, folds, seeds 42 123 456
    python stageB_convedsr_1024_full.py run 42              # one seed
    python stageB_convedsr_1024_full.py install             # after all 300 runs are finished
"""
import os, sys, json, time, glob, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageB_convedsr_1024 as C                      # network, 1024 Hz input, batching (not modified)
B, A = C.B, C.A
SEEDS = (42, 123, 456)
RESF = os.path.join(A.OUT, "stageB_1024full")
os.makedirs(RESF, exist_ok=True)


def run(seeds, log):
    lr = B.fixed_lr("convedsr")
    A.make_batch = C.make_batch
    log(f"=== ConvED-SR, 1024 Hz HG input | lr {lr} | seeds {seeds} | all participants x 10 folds")
    for seed in seeds:
        for sub in A.ALL_SUBS:
            for k in range(A.NBLK):
                tag = f"convedsr_{sub}_fold{k}_seed{seed}"
                if os.path.exists(os.path.join(RESF, tag + ".json")):
                    continue
                t0 = time.time(); B.set_fold(k)
                D, ymu, ysd = A.load_all([sub]); C.prepare(D, sub)
                m, v = A.train_model("convedsr1024", lr, seed, D, A.MAX_EP, lambda s: log(f"  {sub} fold {k} seed {seed}: " + s.strip()))
                wins = A.windows(D, "test")
                P_raw = A.predict(m, D, wins) * ysd[None, :, None] + ymu[None, :, None]
                res, preds = A.evaluate(D, wins, P_raw); r = res[sub]
                np.savez_compressed(os.path.join(RESF, tag + "_preds.npz"), **preds)
                json.dump(dict(model="convedsr", input="hg1024", lr=lr, seed=seed, subject=sub, fold=k, per_subject={sub: r}),
                          open(os.path.join(RESF, tag + ".json"), "w"), indent=1)
                log(f"{tag}: PCC {r['pcc']:.3f} | in speech {r['pcc_in_speech']:.3f} | residual {r['residual_pcc']:+.3f} | {time.time() - t0:.0f}s")


def install():
    done = glob.glob(os.path.join(RESF, "convedsr_*_seed*.json"))
    assert len(done) == 300, f"only {len(done)} of 300 runs finished; run again before installing"
    arc = os.path.join(A.OUT, "archive_convedsr_hgframes")
    for sub_dir in ("stageB", "stageB_seeds_preds"):
        os.makedirs(os.path.join(arc, sub_dir), exist_ok=True)
        for f in glob.glob(os.path.join(A.OUT, sub_dir, "convedsr_*")):
            shutil.move(f, os.path.join(arc, sub_dir))
    n = 0
    for f in done:
        j = json.load(open(f)); s, k, sd = j["subject"], j["fold"], j["seed"]
        src = f[:-5]
        if sd == 42: dst = os.path.join(A.OUT, "stageB", f"convedsr_{s}_fold{k}")
        else: dst = os.path.join(A.OUT, "stageB_seeds_preds", f"convedsr_{s}_fold{k}_seed{sd}")
        shutil.copy(src + ".json", dst + ".json"); shutil.copy(src + "_preds.npz", dst + "_preds.npz"); n += 1
    print(f"installed {n} runs; the HG-frame ConvED-SR files are in {arc}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["install"]:
        install()
    elif args[:1] == ["run"]:
        seeds = [int(a) for a in args[1:]] or list(SEEDS)
        logf = open(os.path.join(RESF, "log_convedsr_1024.txt"), "a")
        def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
        run(seeds, log)
    else:
        raise SystemExit(__doc__)
