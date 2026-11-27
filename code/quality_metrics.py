r"""
quality_metrics.py -- MCD (fast, no vocoder) and STOI/eSTOI (HiFi-GAN, slow on CPU) for Stage A (seed 42) and Stage B (all 10 folds).
Nothing here trains or changes any model; it only reads the saved predictions (<OUT>\stageA\*_preds.npz, <OUT>\stageB\*_preds.npz).
stageA_stoi.py is NOT modified (its Stage-A STOI/eSTOI results stay as they are).

MCD (mel-cepstral distortion), computed on the SAME test frames as every other metric:
  the predicted and the true log-mel (natural log, 80 bins, the HiFi-GAN grid) are turned into mel-cepstra with an orthonormal DCT-II over the
  80 mel bins; coefficients c1..c24 are kept (c0 = energy is excluded, as is usual), and per frame
      MCD = (10 / ln 10) * sqrt( 2 * sum_k (c_k - c'_k)^2 )    [dB],   averaged over frames.   LOWER = better.
  It is reported over all test frames and over speech frames only (same Otsu speech mask as the in-speech PCC).
  NOTE: this is MCD on mel-cepstra derived from the log-mel (no WORLD/SPTK re-analysis of waveforms), so absolute values are not comparable with
  MCD numbers in other papers; it is comparable across our systems (all use exactly the same definition and frames).
  Systems: every model + 'template' (no neural input; the floor).
  Per participant: value = mean over its test folds (Stage B) or over its test blocks (Stage A).

STOI / eSTOI (Stage B): as stageA_stoi.py (SpeechBrain HiFi-GAN of the predicted mel vs the original audio, pystoi), one test block per fold.
  The 'truth' (HiFi-GAN of the true mel) and 'template' systems are scored once per participant-fold.

Usage (SEEG_OUT_DIR set; stageA_pooled.py etc. in the same folder; add the word  smoke  to use the stageB_smoke folder):
    python quality_metrics.py mcd                 # Stage A + Stage B, fast (seconds-minutes)
    python quality_metrics.py stoiB               # Stage B STOI/eSTOI, long on CPU, resumable (finished items skipped)
    python quality_metrics.py stoiB sub-06        # only this participant
    python quality_metrics.py stoiB folds=0       # only fold 0 of every participant (this is also the fold saved as wav for listening)
    python quality_metrics.py stoiB folds=0 sub-06# timing test: one participant, one fold
Listening wavs: <OUT>\stageB_audio\<sub>_fold0_<system>.wav  (systems: original, truth, template, linear, baseline, hybrid, neuroincept, convedsr)
    python quality_metrics.py summary
Results: <OUT>\quality\mcd_results.json  and  <OUT>\quality\stoiB_results.json
"""
import os, sys, glob, json, re, time
import numpy as np

OUT = os.environ["SEEG_OUT_DIR"]
SMOKE = "smoke" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "smoke"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SR, HOP = 22050, 256
QD = os.path.join(OUT, "quality_smoke" if SMOKE else "quality"); os.makedirs(QD, exist_ok=True)
JS_MCD, JS_STOI = os.path.join(QD, "mcd_results.json"), os.path.join(QD, "stoiB_results.json")
DIR_A = os.path.join(OUT, "stageA")
DIR_B = os.path.join(OUT, "stageB_smoke" if SMOKE else "stageB")
ALL_SUBS = [f"sub-{i:02d}" for i in range(1, 11)]
try:
    import pyworld, pysptk; HAVE_WORLD = True
except Exception:
    HAVE_WORLD = False                                      # pip install pyworld pysptk "setuptools<81"
AUD = os.path.join(OUT, "stageB_audio_smoke" if SMOKE else "stageB_audio")      # wavs for listening (Stage B)
LISTEN_FOLD = 0                                                                  # the test block saved as wav for every participant and system
NCEP = 24
MODELS = ["linear", "baseline", "hybrid", "neuroincept", "convedsr"]
LABEL = {"linear": "Linear (Verwoert-style)", "baseline": "Baseline CNN-BiLSTM", "hybrid": "Hybrid (MR-ResBlocks)",
         "neuroincept": "NeuroIncept (re-impl.)", "convedsr": "ConvED-SR-style (re-impl.)", "template": "Template (no neural input)",
         "truth": "HiFi-GAN of true mel (ceiling)"}


# ----------------------------------------------------------------------------------------------- MCD
def _dct_matrix(n=80, k=NCEP + 1):
    """orthonormal DCT-II rows 0..k-1 over n mel bins"""
    i = np.arange(n); m = np.zeros((k, n))
    for c in range(k):
        m[c] = np.cos(np.pi * c * (2 * i + 1) / (2 * n)) * (np.sqrt(1.0 / n) if c == 0 else np.sqrt(2.0 / n))
    return m
DCT = _dct_matrix()


def mcd_frames(pred, true):
    """(T,80),(T,80) natural-log mel -> MCD in dB per frame (c1..c24)"""
    d = (pred @ DCT.T - true @ DCT.T)[:, 1:NCEP + 1]
    return (10.0 / np.log(10.0)) * np.sqrt(2.0 * (d ** 2).sum(1))


def _speech_mask(spec):
    from template_floor_continuous import speech_mask
    return speech_mask(spec)


def mcd_of_file(path, sysnames):
    """-> {sub: {system: (mcd_all, mcd_speech, n_frames)}} for every participant stored in the npz"""
    z = np.load(path); out = {}
    for key in z.files:
        if not key.endswith("_idx"):
            continue
        sub = key[:-4]
        spec = np.load(os.path.join(OUT, f"{sub}_spec.npy")); idx = z[key]
        sp = _speech_mask(spec)[idx]; true = spec[idx]
        out[sub] = {}
        for s in sysnames:
            P = z[sub + ("_T" if s == "template" else "_P")]
            f = mcd_frames(P.astype(np.float64), true.astype(np.float64))
            out[sub][s] = (float(f.mean()), float(f[sp].mean()) if sp.any() else float("nan"), int(len(f)))
    return out


def run_mcd():
    res = json.load(open(JS_MCD)) if os.path.exists(JS_MCD) else {}
    # ---- Stage A: seed 42 (linear has no lr/seed)
    for m in MODELS:
        f = sorted(glob.glob(os.path.join(DIR_A, f"{m}_lr*_seed42_preds.npz"))) + sorted(glob.glob(os.path.join(DIR_A, f"{m}_seedNA_preds.npz")))
        if not f:
            continue
        systems = [m] + (["template"] if m == "hybrid" else [])
        for sub, d in mcd_of_file(f[0], systems).items():
            for s, (a, b, n) in d.items():
                res[f"A|{s}|{sub}|-"] = dict(stage="A", system=s, subject=sub, fold=-1, mcd=a, mcd_speech=b, frames=n)
    # ---- Stage B: every finished participant-fold
    for p in sorted(glob.glob(os.path.join(DIR_B, "*_preds.npz"))):
        mm = re.match(r"(.+?)_(sub-\d+)_fold(\d+)_preds\.npz", os.path.basename(p))
        if not mm:
            continue
        m, sub, k = mm.group(1), mm.group(2), int(mm.group(3))
        systems = [m] + (["template"] if m == "hybrid" else [])
        d = mcd_of_file(p, systems)[sub]
        for s, (a, b, n) in d.items():
            res[f"B|{s}|{sub}|{k}"] = dict(stage="B", system=s, subject=sub, fold=k, mcd=a, mcd_speech=b, frames=n)
    json.dump(res, open(JS_MCD, "w"), indent=1)
    print(f"MCD: {len(res)} entries written -> {JS_MCD}")


# ----------------------------------------------------------------------------------------------- waveform MCD (WORLD + SPTK), as in ConvED-SR eq. (20)
def _mcep(x, fs=SR, order=24, hopms=5.0):
    import pyworld as pw, pysptk
    f0, t = pw.dio(x, fs, frame_period=hopms); f0 = pw.stonemask(x, f0, t, fs)
    return pysptk.sp2mc(pw.cheaptrick(x, f0, t, fs), order=order, alpha=pysptk.util.mcepalpha(fs))[:, 1:]       # c1..c24 (c0 dropped)


def mcd_wave(ref, deg, fs=SR, hopms=5.0):
    """MCD (dB) between two waveforms of the same length: (10/ln10)*sqrt(2*sum_n (c_n - c'_n)^2), n = 1..24, mean over frames
    (all frames, and frames where the reference energy is > 5% of its maximum = 'active')"""
    n = min(len(ref), len(deg)); ref, deg = ref[:n].astype(np.float64), deg[:n].astype(np.float64)
    a, b = _mcep(ref, fs, hopms=hopms), _mcep(deg, fs, hopms=hopms); m = min(len(a), len(b))
    f = (10.0 / np.log(10.0)) * np.sqrt(2.0 * ((a[:m] - b[:m]) ** 2).sum(1))
    hop, w = int(hopms * fs / 1000), int(0.0125 * fs)
    r = np.array([np.sqrt(np.mean(ref[max(0, i * hop - w): i * hop + w] ** 2) + 1e-12) for i in range(m)])
    act = r > 0.05 * r.max()
    return float(f.mean()), float(f[act].mean()) if act.any() else float("nan")


# ----------------------------------------------------------------------------------------------- STOI / eSTOI, Stage B
def run_stoiB(subs, folds=None):
    import stageA_stoi as S                                  # re-uses the vocoder / wav reader / run splitter (unchanged file)
    from pystoi import stoi
    from scipy.io import wavfile
    os.makedirs(AUD, exist_ok=True)
    decode = S.get_vocoder()
    res = json.load(open(JS_STOI)) if os.path.exists(JS_STOI) else {}
    for sub in subs:
        wav = glob.glob(os.path.join(OUT, f"{sub}_audio_*.wav"))
        if not wav:
            print("no audio for", sub); continue
        sr, x = S.read_wav(wav[0]); assert sr == SR
        spec = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
        files = sorted(glob.glob(os.path.join(DIR_B, f"*_{sub}_fold*_preds.npz")))
        for p in files:
            mm = re.match(r"(.+?)_(sub-\d+)_fold(\d+)_preds\.npz", os.path.basename(p)); m, k = mm.group(1), int(mm.group(3))
            if folds is not None and k not in folds:
                continue
            z = np.load(p); idx = z[sub + "_idx"]
            todo = [(m, z[sub + "_P"])]
            if m == "hybrid":                                 # truth + template once per participant-fold (taken from the hybrid file)
                todo += [("template", z[sub + "_T"]), ("truth", spec[idx])]
            for s, mel in todo:
                key = f"B|{s}|{sub}|{k}"
                if key in res and (not HAVE_WORLD or "mcd_world" in res[key]):
                    continue
                t0 = time.time(); v, e, mw, mwa = [], [], [], []
                for a, b in S.runs_of(idx):
                    fr0, fr1 = int(idx[a]), int(idx[b - 1]) + 1
                    y = decode(mel[a:b]); ref = x[fr0 * HOP: fr1 * HOP]; n = min(len(ref), len(y))
                    v.append(stoi(ref[:n], y[:n], SR, extended=False)); e.append(float("nan"))   # eSTOI is not used in the paper and is not computed
                    if HAVE_WORLD:
                        q = mcd_wave(ref[:n], y[:n]); mw.append(q[0]); mwa.append(q[1])
                    if k == LISTEN_FOLD and (a, b) == S.runs_of(idx)[0]:                    # listening files: first run of the listening fold
                        w16 = lambda v_: (v_ / (np.abs(v_).max() + 1e-9) * 0.9 * 32767).astype(np.int16)
                        wavfile.write(os.path.join(AUD, f"{sub}_fold{k}_{s}.wav"), SR, w16(y[:n]))
                        if s == "truth":
                            wavfile.write(os.path.join(AUD, f"{sub}_fold{k}_original.wav"), SR, w16(ref[:n]))
                res[key] = dict(stage="B", system=s, subject=sub, fold=k, stoi=float(np.mean(v)), estoi=float(np.mean(e)))
                if HAVE_WORLD:
                    res[key].update(mcd_world=float(np.mean(mw)), mcd_world_active=float(np.nanmean(mwa)))
                json.dump(res, open(JS_STOI, "w"), indent=1)
                print(f"  {key}: STOI {np.mean(v):.3f} | eSTOI {np.mean(e):.3f} | {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- summary
def _paired(df, metric, a, b):
    from scipy.stats import wilcoxon
    t = df.pivot(index="subject", columns="system", values=metric)[[a, b]].dropna()
    if len(t) < 5:
        return None
    d = t[a] - t[b]
    p = wilcoxon(d, method="exact").pvalue if (d != 0).any() else 1.0
    return len(t), float(d.mean()), int((d < 0).sum()) if metric in ("mcd", "mcd_speech") else int((d > 0).sum()), float(p)


def summary():
    import pandas as pd
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    if os.path.exists(JS_MCD):
        df = pd.DataFrame(json.load(open(JS_MCD)).values())
        for stage, title in (("A", "STAGE A (pooled, seed 42)"), ("B", "STAGE B (subject-dependent, mean over folds)")):
            d = df[df.stage == stage]
            if d.empty: continue
            pp = d.groupby(["subject", "system"])[["mcd", "mcd_speech"]].mean().reset_index()
            pp["system"] = pp["system"].map(lambda s: LABEL.get(s, s))
            print(f"\n==== MCD in dB, LOWER = better | {title}")
            for met in ("mcd", "mcd_speech"):
                t = pp.pivot(index="subject", columns="system", values=met)
                t.loc["mean"] = t.mean()
                print(f"\n{'all test frames' if met == 'mcd' else 'speech frames only'}:"); print(t.round(2).to_string())
            sysd = d.groupby(["subject", "system"])[["mcd", "mcd_speech"]].mean().reset_index()
            for met in ("mcd", "mcd_speech"):
                rows = []
                for a, b in (("hybrid", "baseline"), ("hybrid", "linear"), ("hybrid", "convedsr"), ("hybrid", "neuroincept"), ("hybrid", "template")):
                    if a in set(sysd.system) and b in set(sysd.system):
                        r = _paired(sysd, met, a, b)
                        if r: rows.append(dict(comparison=f"{LABEL[a]} minus {LABEL[b]}", metric=met, n=r[0], mean_diff=round(r[1], 3), a_better_in=f"{r[2]}/{r[0]}", p_exact=round(r[3], 4)))
                if rows: print(f"\npaired ({met}; negative difference = first system better):"); print(pd.DataFrame(rows).to_string(index=False))
    if os.path.exists(JS_STOI):
        ds = pd.DataFrame(json.load(open(JS_STOI)).values())
        pp = ds.groupby(["subject", "system"])[["stoi", "estoi"]].mean().reset_index(); pp["system"] = pp["system"].map(lambda s: LABEL.get(s, s))
        print("\n==== STOI / eSTOI, Stage B (mean over folds), HIGHER = better")
        for met in ("stoi", "estoi"):
            t = pp.pivot(index="subject", columns="system", values=met); t.loc["mean"] = t.mean()
            print(f"\n{met.upper()}:"); print(t.round(3).to_string())
        if "mcd_world" in ds.columns:
            for met, ttl in (("mcd_world", "all frames"), ("mcd_world_active", "active (speech) frames")):
                pw_ = ds.dropna(subset=[met]).groupby(["subject", "system"])[met].mean().reset_index(); pw_["system"] = pw_["system"].map(lambda s_: LABEL.get(s_, s_))
                t = pw_.pivot(index="subject", columns="system", values=met); t.loc["mean"] = t.mean()
                print(f"\nMCD in dB (WORLD+SPTK, c1-c24, ConvED-SR eq. 20), LOWER = better | {ttl}:"); print(t.round(2).to_string())
        n = ds.groupby("system")["fold"].count()
        print("\nfinished participant-folds per system:", n.to_dict())


if __name__ == "__main__":
    if ARGS[:1] == ["mcd"]:
        run_mcd(); summary()
    elif ARGS[:1] == ["stoiB"]:
        fo = [a for a in ARGS[1:] if a.startswith("folds=")]
        run_stoiB([a for a in ARGS[1:] if a.startswith("sub-")] or ALL_SUBS, [int(x) for x in fo[0][6:].split(",")] if fo else None); summary()
    elif ARGS[:1] == ["summary"]:
        summary()
    else:
        print(__doc__)
