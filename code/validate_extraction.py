"""
validate_extraction.py  --  checks for the continuous 80-bin extraction (hifigan mode)

Usage (terminal, NOT inside a notebook cell that you also use for other things):
    set SEEG_OUT_DIR=D:\\seeg_features_hifigan        (Windows)   /  export SEEG_OUT_DIR=...
    python validate_extraction.py            # checks 1-3 for all subjects (fast, no torch)
    python validate_extraction.py ceiling    # check 4: HiFi-GAN ceiling STOI (needs speechbrain, torch, pystoi)

Checks
 1. Shapes / NaN / inf / constant channels, T identical for feat, spec, frame_times, labels
 2. Neural-acoustic lag: cross-correlation of channel-mean HG envelope vs mel energy (-500..+500 ms)
 3. Plot: HG mean vs mel energy for 20 s (saved as PNG, look at it)
 4. Ceiling: STOI(original audio, HiFi-GAN(true mel)) -- the upper bound for any STOI you report
"""
import os, sys, glob, json
import numpy as np
from scipy.io import wavfile

OUT = os.environ.get("SEEG_OUT_DIR", ".")
HOP = 256
SR = 22050
FRAME_S = HOP / SR  # 11.61 ms


def subjects():
    return sorted(os.path.basename(p)[:6] for p in glob.glob(os.path.join(OUT, "sub-*_feat.npy")))


def load(sub):
    f = np.load(os.path.join(OUT, f"{sub}_feat.npy"))
    s = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
    t = np.load(os.path.join(OUT, f"{sub}_frame_times.npy"))
    return f, s, t


def check_basic(sub):
    f, s, t = load(sub)
    ok = True
    msgs = []
    if not (len(f) == len(s) == len(t)):
        ok = False; msgs.append(f"LENGTH MISMATCH feat {len(f)} spec {len(s)} times {len(t)}")
    if s.shape[1] != 80:
        ok = False; msgs.append(f"mel bins {s.shape[1]} != 80")
    for name, a in (("feat", f), ("spec", s)):
        if not np.isfinite(a).all():
            ok = False; msgs.append(f"{name}: NaN/inf = {(~np.isfinite(a)).sum()}")
    const = int((f.std(0) < 1e-12).sum())
    if const:
        ok = False; msgs.append(f"{const} constant channels")
    dt = np.diff(t)
    if np.abs(dt - FRAME_S).max() > 1e-6:
        msgs.append(f"frame step not constant (min {dt.min():.6f}, max {dt.max():.6f}); expected {FRAME_S:.6f}")
    print(f"{sub}: frames={len(f)} ch={f.shape[1]} mel={s.shape} dur={t[-1]:.1f}s "
          f"spec[min,max]=({s.min():.2f},{s.max():.2f}) -> {'OK' if ok else 'PROBLEM'}")
    for m in msgs:
        print("    !", m)
    return ok


def _highpass(x, win):
    """subtract a moving average of `win` frames along axis 0 (removes slow drifts that dominate raw correlations)"""
    k = np.ones(win) / win
    if x.ndim == 1:
        return x - np.convolve(x, k, mode="same")
    return x - np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)


def check_lag(sub, maxlag_ms=500, plot=True, topk=10):
    """Per-channel lag analysis. The channel-AVERAGE is a poor test: most contacts carry no speech information,
    so averaging dilutes the signal and slow drifts decide the peak. Here every channel is correlated with the
    mel energy (drift removed with a 2 s moving average), the topk best channels are kept and their peak lags reported."""
    f, s, t = load(sub)
    W = int(round(2.0 / FRAME_S))
    X = _highpass(f.astype(np.float64), W)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    en = _highpass(s.mean(1).astype(np.float64), W)
    en = (en - en.mean()) / (en.std() + 1e-9)
    n = int(round(maxlag_ms / 1000 / FRAME_S))
    lags = np.arange(-n, n + 1)
    cc = np.zeros((len(lags), X.shape[1]))
    for i, L in enumerate(lags):  # positive L: neural at time t correlates with audio at t+L (neural leads)
        if L >= 0:
            a, b = X[: len(X) - L], en[L:]
        else:
            a, b = X[-L:], en[: len(en) + L]
        cc[i] = (a * b[:, None]).mean(0)
    pk = cc.max(0); pk_lag = lags[cc.argmax(0)] * FRAME_S * 1000
    order = np.argsort(pk)[::-1][:topk]
    med = np.median(pk_lag[order])
    n_edge = int((np.abs(pk_lag[order]) >= maxlag_ms - 15).sum())
    c0 = cc[n, order].mean()
    print(f"  {sub}: top-{topk} channels  corr {pk[order].mean():.3f} (at lag 0: {c0:.3f}) | "
          f"peak lags ms: median {med:+.0f}, range [{pk_lag[order].min():+.0f}, {pk_lag[order].max():+.0f}]"
          + (f" | {n_edge} at the search edge (unreliable)" if n_edge else ""))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        b = order[0]
        m = (t > 60) & (t < 80)
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax[0].plot(t[m], X[m, b]); ax[0].set_ylabel(f"HG ch {b} (z)")
        ax[1].plot(t[m], en[m]); ax[1].set_ylabel("mean mel (z)"); ax[1].set_xlabel("s")
        fig.suptitle(f"{sub}  best channel {b}: peak lag {pk_lag[b]:+.0f} ms, corr {pk[b]:.2f}")
        fig.savefig(os.path.join(OUT, f"{sub}_check_hg_vs_mel.png"), dpi=110)
        plt.close(fig)
    return med


def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    return sr, x


def ceiling(segment_s=30.0):
    import torch
    from speechbrain.inference.vocoders import HIFIGAN
    from pystoi import stoi
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {}
    try:                                    # Windows: symlinks need admin rights -> copy the files instead
        from speechbrain.utils.fetching import LocalStrategy
        kw["local_strategy"] = LocalStrategy.COPY
    except Exception:
        pass
    hifi = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                savedir=os.path.join(OUT, "hifigan_ckpt"), run_opts={"device": dev}, **kw)
    res = {}
    for sub in subjects():
        s = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
        wav_path = glob.glob(os.path.join(OUT, f"{sub}_audio_*.wav"))[0]
        sr, x = read_wav(wav_path)
        assert sr == SR, f"audio sr {sr} != {SR}"
        vals = []
        nfr = int(segment_s * SR / HOP)
        # three segments: start, middle, end (each ~30 s)
        for a in (0, len(s) // 2 - nfr // 2, len(s) - nfr - 1):
            a = max(a, 0); b = min(a + nfr, len(s))
            mel = torch.tensor(s[a:b].T, dtype=torch.float32)[None].to(dev)  # (1,80,T)
            with torch.no_grad():
                y = hifi.decode_batch(mel).squeeze().cpu().numpy()
            ref = x[a * HOP: b * HOP]
            n = min(len(ref), len(y))
            vals.append(stoi(ref[:n], y[:n], SR, extended=False))
        res[sub] = float(np.mean(vals))
        print(f"  {sub}: ceiling STOI (HiFi-GAN of TRUE mel vs original audio) = {res[sub]:.3f}  segments {np.round(vals,3)}")
    print("mean ceiling STOI over subjects: %.3f" % np.mean(list(res.values())))
    json.dump(res, open(os.path.join(OUT, "hifigan_ceiling_stoi.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ceiling":
        ceiling()
    else:
        subs = subjects()
        print("subjects found:", subs)
        ok = all([check_basic(s) for s in subs])
        print("\nneural-acoustic lag:")
        lags = [check_lag(s) for s in subs]
        print("\nALL BASIC CHECKS PASSED" if ok else "\nSOME CHECKS FAILED -- do not train yet")
