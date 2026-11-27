"""
stageB_convedsr_1024.py -- OPTIONAL CHECK (not in the paper): ConvED-SR re-implementation fed with 1024 Hz sEEG instead of 50 ms HG frames.

Question answered: is the low score of our ConvED-SR re-implementation caused by giving it HG frames (one value per mel frame) rather than
1024 Hz samples? Everything else is identical to stageB_subject.py (same folds, windows, loss, optimiser, scheduler, early stopping,
seed 42, learning rate of ConvED-SR from Stage A, metrics, template). Imports stageB_subject / stageA_extra, which are NOT modified.

What changes: each 40-frame segment (464 ms) is no longer 40 HG frames but the 475 samples (464 ms x 1024 Hz) of the same time span.
The Conv1d stem (stride 2) therefore sees 475 instead of 40 steps; the last encoder kernel becomes (2,2) instead of (2,1)
(it is computed from the input length). Two input variants (environment variable CONVED_INPUT):
    hg1024  (default) : the same HG envelope as in the paper (detrend, 70-170 Hz band-pass, line-noise band-stops, |Hilbert|) at 1024 Hz,
                        WITHOUT the 50 ms averaging and WITHOUT down-sampling
    raw               : the broadband sEEG samples (as stored in the dataset), no filtering
In both, every channel is z-scored with the training blocks of the fold only.

Needs: SEEG_OUT_DIR (the features folder, as for the other scripts) and SEEG_DATA_ROOT (the dataset folder with the .nwb files), and
pynwb installed (already used by extract_continuous_80.py).

Usage (all scripts in the same folder):
    python stageB_convedsr_1024.py smoke                  # 1 epoch, 1 participant x 1 fold: checks the plumbing and prints seconds per epoch
    python stageB_convedsr_1024.py run                    # sub-02, sub-06, sub-10 x paper folds 1 and 6 (code folds 0 and 5): 6 runs
    python stageB_convedsr_1024.py run sub-02 0 5         # only this participant and these (code) folds
    python stageB_convedsr_1024.py summary                # table: main ConvED-SR (HG frames) versus this run, same participant-fold
Runs are resumable. Results: <SEEG_OUT_DIR>/stageB_1024/<variant>_<sub>_fold<k>.json
Optional: STAGEB_THREADS=4 limits the CPU threads.
"""
import os, sys, json, time, glob
import numpy as np
import torch, torch.nn.functional as F

if os.environ.get("STAGEB_THREADS"):
    torch.set_num_threads(int(os.environ["STAGEB_THREADS"]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SMOKE = sys.argv[1:2] == ["smoke"]
sys.argv = [sys.argv[0]] + (["smoke"] if SMOKE else []) + [a for a in sys.argv[1:] if a != "smoke"]   # let stageA_pooled see "smoke" as argv[1]
import stageB_subject as B
import stageA_extra as E
import extract_continuous_80 as X
A = B.A

VARIANT = os.environ.get("CONVED_INPUT", "hg1024")
assert VARIANT in ("hg1024", "raw")
SR, MODEL = 1024, "convedsr"
LSEG = int(round(E.ConvEDSR.SEG * 256 / 22050 * SR))          # 475 samples = 40 frames
RES1 = os.path.join(A.OUT, "stageB_1024_smoke" if SMOKE else "stageB_1024")
os.makedirs(RES1, exist_ok=True)


class ConvEDSR1024(E.ConvEDSR):
    """same network; only the input length of a 40-frame segment changes (475 samples instead of 40 frames)"""
    def _final_width(self):
        w = (LSEG + 2 - 3) // 2 + 1
        for _ in range(7):
            w = (w + 2 - 3) // 2 + 1
        return w

    def _segments(self, x):                                   # x (B, CH, S, LSEG) -> (B*S, CH, LSEG)
        return x.permute(0, 2, 1, 3).flatten(0, 1)


A.CTORS["convedsr1024"] = ConvEDSR1024


def raw_signal(sub):
    """full-rate (Ns, C) float32 signal of the recording, cached on disk"""
    f = os.path.join(A.OUT, f"cache1024_{VARIANT}_{sub}.npy")
    if os.path.exists(f):
        return np.load(f)
    from pynwb import NWBHDF5IO
    p = os.path.join(os.environ["SEEG_DATA_ROOT"], sub, "ieeg", f"{sub}_task-wordProduction_ieeg.nwb")
    io = NWBHDF5IO(p, "r"); eeg = np.asarray(io.read().acquisition["iEEG"].data[:], dtype=np.float64); io.close()
    sig = X.hg_envelope(eeg, SR) if VARIANT == "hg1024" else eeg
    sig = sig.astype(np.float32); np.save(f, sig)
    return sig


def prepare(D, sub):
    """adds the full-rate signal, normalised with the TRAIN blocks of the current fold, and the sample start of every frame"""
    d = D[sub]
    ft = np.load(os.path.join(A.OUT, f"{sub}_frame_times.npy"))
    sig = raw_signal(sub)
    assert sig.shape[1] == d["feat"].shape[1], f"{sub}: {sig.shape[1]} channels in the signal, {d['feat'].shape[1]} in the features"
    half = 0.5 * 256 / 22050
    st = np.round((ft - half) * SR).astype(np.int64)             # first sample of the span of each frame
    keep = np.zeros(len(sig), bool)
    for b, (s, e) in enumerate(d["blocks"]):
        if b in A.TEST_B + A.VAL_B: continue
        keep[max(st[s], 0):min(st[e - 1] + int(round(256 / 22050 * SR)), len(sig))] = True
    mu, sd = sig[keep].mean(0), sig[keep].std(0) + 1e-6
    d["Xraw"] = np.concatenate([((sig - mu) / sd).astype(np.float32), np.zeros((2 * LSEG, sig.shape[1]), np.float32)])   # zero tail so slices never run short
    d["st_samp"] = np.clip(st, 0, len(sig) - 1)


def make_batch(D, wins, idx):
    S = len(E.ConvEDSR.STARTS); Bn = len(idx); CH = A.CH
    x = np.zeros((Bn, CH, S, LSEG), np.float32); m = np.zeros((Bn, CH, 1, 1), np.float32)
    y = np.zeros((Bn, A.MEL, A.W), np.float32); sid = np.zeros(Bn, np.int64)
    for j, i in enumerate(idx):
        sub, st = wins[i]; d = D[sub]; c = d["Xraw"].shape[1]
        for k, s in enumerate(E.ConvEDSR.STARTS):
            a = d["st_samp"][st + s]
            x[j, :c, k] = d["Xraw"][a:a + LSEG].T
        m[j, :c] = 1; y[j] = d["Yn"][st:st + A.W].T; sid[j] = d["sid"]
    return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y), torch.from_numpy(sid)


def run(subs, folds, log):
    lr = B.fixed_lr(MODEL)
    log(f"=== ConvED-SR with {SR} Hz input ({VARIANT}) | lr {lr} | participants {subs} | folds {folds}")
    A.make_batch = make_batch                                   # patched here only; stageA_pooled.py itself is untouched
    for sub in subs:
        for k in folds:
            tag = f"{VARIANT}_{sub}_fold{k}"
            if os.path.exists(os.path.join(RES1, tag + ".json")):
                continue
            t0 = time.time(); B.set_fold(k)
            D, ymu, ysd = A.load_all([sub]); prepare(D, sub)
            m, v = A.train_model("convedsr1024", lr, B.SEED, D, 1 if SMOKE else A.MAX_EP, lambda s: log(f"  {sub} fold {k}: " + s.strip()))
            wins = A.windows(D, "test")
            P_raw = A.predict(m, D, wins) * ysd[None, :, None] + ymu[None, :, None]
            res, _ = A.evaluate(D, wins, P_raw); r = res[sub]
            json.dump(dict(model="convedsr_1024Hz", variant=VARIANT, lr=lr, seed=B.SEED, subject=sub, fold=k, per_subject={sub: r}),
                      open(os.path.join(RES1, tag + ".json"), "w"), indent=1)
            log(f"{tag}: PCC {r['pcc']:.3f} | in speech {r['pcc_in_speech']:.3f} | residual {r['residual_pcc']:+.3f} | {time.time() - t0:.0f}s")


def summary():
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(RES1, f"{VARIANT}_*.json"))):
        j = json.load(open(f)); sub, k = j["subject"], j["fold"]; r = j["per_subject"][sub]
        mf = os.path.join(A.OUT, "stageB", f"convedsr_{sub}_fold{k}.json")
        if not os.path.exists(mf): continue
        m = json.load(open(mf))["per_subject"][sub]
        rows.append(dict(subject=sub, fold_paper=k + 1, pcc_HGframes=m["pcc"], pcc_1024Hz=r["pcc"], in_speech_HGframes=m["pcc_in_speech"],
                         in_speech_1024Hz=r["pcc_in_speech"], residual_HGframes=m["residual_pcc"], residual_1024Hz=r["residual_pcc"]))
    df = pd.DataFrame(rows)
    if df.empty: print("no finished runs yet"); return
    pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
    print("\nmean over the", len(df), "participant-folds:"); print(df.drop(columns=["subject", "fold_paper"]).mean().round(3).to_string())
    df.to_csv(os.path.join(A.OUT, f"stageB_1024_{VARIANT}_summary.csv"), index=False)


if __name__ == "__main__":
    def log(s): print(s, flush=True)
    args = [a for a in sys.argv[1:] if a != "smoke"]
    if args[:1] == ["summary"]:
        summary()
    elif SMOKE:
        run(["sub-02"], [0], log)
    elif args[:1] == ["run"]:
        if len(args) > 1:
            run([args[1]], [int(a) for a in args[2:]] or [0, 5], log)
        else:
            run(["sub-02", "sub-06", "sub-10"], [0, 5], log)
    else:
        raise SystemExit(__doc__)
