# selftest_extraction.py -- run with:  python extract_continuous_80.py selftest   (needs /mnt/user-data/uploads/extract_features.py
# or set VERWOERT_SCRIPT to the path of Verwoert's original extract_features.py). Synthetic data only; checks the code against the
# ORIGINAL Verwoert functions and checks alignment.
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
UP = os.environ.get("VERWOERT_SCRIPT", os.path.join(_here, "extract_features.py"))   # Verwoert's ORIGINAL file, copy it next to this script
if not os.path.exists(UP):
    raise SystemExit(f"Verwoert's original extract_features.py not found at {UP}. Copy it into this folder (or set VERWOERT_SCRIPT).")
src = open(UP, encoding="utf-8", errors="replace").read()
ns = {"np": np, "scipy": scipy, "MelFilterBank": MelFilterBank}                    # original code, np.float/scipy.hanning patched
ns["hilbert3"] = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis=0)[:len(x)]
exec("def extractHG" + src.split("def extractHG")[1].split("def stackFeatures")[0], ns)
m_src = "def extractMelSpecs" + src.split("def extractMelSpecs")[1].split("def nameVector")[0]
m_src = m_src.replace("scipy.hanning(np.floor(windowLength*sr + 1))[:-1]", "scipy.signal.windows.hann(int(np.floor(windowLength*sr + 1)), sym=True)[:-1]")
m_src = m_src.replace("mel.MelFilterBank(", "MelFilterBank(")
exec(m_src, ns)

rng = np.random.RandomState(0)
sr = 1024; N = sr * 40; C = 6
eeg0 = rng.randn(N, C).cumsum(axis=0) * 0.01 + rng.randn(N, C)
ref = ns["extractHG"](eeg0, sr)
mine, ok = window_means(hg_envelope(eeg0, sr), sr, 0.01 * np.arange(ref.shape[0]))
assert ok.all() and np.allclose(mine, ref, rtol=1e-9, atol=1e-9), "HG features differ from Verwoert's extractHG"
print("1) HG envelope + 50 ms window means == Verwoert's extractHG:", ref.shape)

a16 = (rng.randn(16000 * 20) * 3000).astype(np.float64)
ref_mel = ns["extractMelSpecs"](a16, 16000)                                       # 23 filters in the original
assert np.allclose(mel_verwoert(a16, n_mel=23), ref_mel, rtol=1e-9, atol=1e-9), "mel code differs from Verwoert's"
print("2) mel code with 23 filters == Verwoert's extractMelSpecs:", ref_mel.shape, "| with 80 filters:", mel_verwoert(a16).shape)

# 3) alignment: neural envelope LEADS the audio by 150 ms; the diagnostic must recover +150 ms in both modes
dur, a_sr = 60, 48000
t_e, t_a = np.arange(dur * sr) / sr, np.arange(dur * a_sr) / a_sr
slow = scipy.signal.sosfiltfilt(scipy.signal.butter(2, 1.5, fs=1000, output='sos'), np.abs(rng.randn(dur * 1000)))
env_t = np.interp(t_e, np.arange(dur * 1000) / 1000, slow); env_t = (env_t / env_t.max()) ** 2
lag_true = 0.15
neural = (rng.randn(dur * sr, 4) * (0.2 + 3 * np.interp(t_e + lag_true, t_e, env_t, right=0)[:, None]))
neural = scipy.signal.sosfiltfilt(scipy.signal.butter(4, [80, 140], btype='band', fs=sr, output='sos'), neural, axis=0)
audio = rng.randn(dur * a_sr) * np.interp(t_a, t_e, env_t)
labels = np.where(env_t > 0.3, "word", "")

def stub_mel_hifigan(a):                                                          # numpy mimic of a center=True, hop 256 log-mel (shape/centre only)
    pad = np.pad(a, 512, mode="reflect"); T = 1 + len(a) // 256
    fr = np.stack([pad[i * 256:i * 256 + 1024] * np.hanning(1024) for i in range(T)])
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    return np.log(np.stack([b.mean(axis=1) for b in np.array_split(p[:, :512], 80, axis=1)], axis=1) + 1e-5)

for mode in ("verwoert", "hifigan"):
    r = process_recording(eeg=neural, eeg_sr=sr, audio=audio, audio_sr=a_sr, labels=labels, mode=mode, mel_fn_hifigan=stub_mel_hifigan)
    step = float(np.median(np.diff(r['frame_times']))); lag, c = lag_diagnostic(r['feat'], r['spec'], step)
    assert r['feat'].shape[0] == r['spec'].shape[0] == len(r['frame_times']) == len(r['labels']) and r['spec'].shape[1] == 80
    assert abs(lag - lag_true) < 0.04, f"{mode}: recovered lag {lag:.3f} s, expected {lag_true}"
    print(f"3) mode {mode}: {r['feat'].shape[0]} frames, step {step * 1000:.2f} ms, recovered neural->audio lag {lag * 1000:+.0f} ms (true +150), corr {c:.2f}")
print("self-test passed")
