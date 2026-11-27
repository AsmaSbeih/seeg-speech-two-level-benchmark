# =====================================================================
# extract_continuous_80.py -- Verwoert-style feature extraction on the WHOLE continuous recording, 80 mel bins.
#
# What is identical to Verwoert et al. (extract_features.py): linear detrend; 4th-order Butterworth band-pass 70-170 Hz
# (zero-phase, sos); band-stops 98-102 and 148-152 Hz; |Hilbert| with next_fast_len; mean of the envelope over a 50 ms window whose
# start index is floor(start_time * sr) and whose stop index is floor(start + 0.05 * sr); the whole recording is filtered at once
# (no per-trial filtering, so no edge transients); audio is decimated/resampled once, not per trial.
# What is different on purpose: 80 mel bins, and (mode "hifigan") the frame grid of the vocoder used for the STOI evaluation.
#
# MODES
#   "verwoert": 10 ms frame shift, 50 ms window, audio 16 kHz, Verwoert's own mel filterbank/spectrogram code (80 filters).
#               Frame i covers [0.01 i, 0.01 i + 0.05] in BOTH the sEEG and the audio => identical alignment to the original.
#   "hifigan":  mel computed exactly as the model card of speechbrain/tts-hifigan-ljspeech prescribes (22.05 kHz, n_fft 1024,
#               hop 256, 80 mels, fmin 0, fmax 8000, slaney, power 1, compression) so the SAME mel can be fed to that vocoder.
#               Frame i is centred at t_i = 256 i / 22050 s (torchaudio center=True); the sEEG window of frame i is
#               [t_i - 0.025, t_i + 0.025], so both are centred on the same instant. Frame shift = 11.61 ms.
#
# OUTPUT per participant (OUT_DIR/sub-XX_*): feat.npy (T, C) float32 un-stacked high-gamma envelope; spec.npy (T, 80) float32;
# labels.npy (T,) str (stimulus label at the frame centre); frame_times.npy (T,) s; audio_<sr>.wav (float32 peak-normalised);
# channels.npy; meta.json.  No context stacking is applied here (ridge/NeuroIncept stack on the fly; CNN/RNN models do not need it).
#
# Usage:  set SEEG_DATA_ROOT and SEEG_OUT_DIR (environment variables) and run  python extract_continuous_80.py hifigan
#         (or "verwoert").  Self-test with synthetic data:  python extract_continuous_80.py selftest
# =====================================================================
import os, sys, json, math
import numpy as np
import scipy.signal, scipy.fftpack, scipy.io.wavfile

WIN_SEC = 0.05
MEL_BINS = 80

# ---------------- Verwoert's mel filterbank (copied; np.float -> float so it runs on current numpy) ----------------
class MelFilterBank():
    def __init__(self, specSize, numCoefficients, sampleRate):
        numBands = int(numCoefficients)
        minMel = 0
        maxMel = self.freqToMel(sampleRate / 2.0)
        melStep = (maxMel - minMel) / (numBands + 1)
        melFilterEdges = np.arange(0, numBands + 2) * melStep
        centerIndices = list(map(lambda x: self.freqToBin(math.floor(self.melToFreq(x)), sampleRate, specSize), melFilterEdges))
        filterMatrix = np.zeros((numBands, specSize))
        for i in range(numBands):
            start, center, end = centerIndices[i:i + 3]
            k1 = float(center - start)
            k2 = float(end - center)
            up = (np.array(range(start, center)) - start) / k1
            down = (end - np.array(range(center, end))) / k2
            filterMatrix[i][start:center] = up
            filterMatrix[i][center:end] = down
        self.melMatrix = filterMatrix.transpose()
        self.melMatrix = self.makeNormal(self.melMatrix / self.normSum(self.melMatrix))
    def normSum(self, x):
        retSum = np.sum(x, axis=0)
        retSum[np.where(retSum == 0)] = 1.0
        return retSum
    def fuzz(self, x):
        return x + 0.0000001
    def freqToBin(self, freq, sampleRate, specSize):
        return int(math.floor((freq / (sampleRate / 2.0)) * specSize))
    def freqToMel(self, freq):
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    def melToFreq(self, mel):
        return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)
    def toMelScale(self, spectrogram):
        return np.dot(spectrogram, self.melMatrix)
    def makeNormal(self, x):
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x
    def toLogMels(self, spectrogram):
        return self.makeNormal(np.log(self.fuzz(self.toMelScale(spectrogram))))

# ---------------- high-gamma envelope of the whole recording (Verwoert) ----------------
def hg_envelope(data, sr):
    """data (samples, channels) -> |analytic| of the band-passed, notch-filtered signal, same shape (as in extractHG)."""
    data = scipy.signal.detrend(data, axis=0)
    sos = scipy.signal.iirfilter(4, [70 / (sr / 2), 170 / (sr / 2)], btype='bandpass', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    sos = scipy.signal.iirfilter(4, [98 / (sr / 2), 102 / (sr / 2)], btype='bandstop', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    sos = scipy.signal.iirfilter(4, [148 / (sr / 2), 152 / (sr / 2)], btype='bandstop', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    return np.abs(scipy.signal.hilbert(data, scipy.fftpack.next_fast_len(len(data)), axis=0)[:len(data)])

def window_means(envelope, sr, start_times, win=WIN_SEC):
    """mean of the envelope over [floor(t*sr), floor(floor(t*sr) + win*sr)) for every start time t (exactly Verwoert's indexing)."""
    cs = np.vstack([np.zeros((1, envelope.shape[1])), np.cumsum(envelope, axis=0, dtype=np.float64)])
    start = np.floor(np.asarray(start_times) * sr).astype(np.int64)
    stop = np.floor(start + win * sr).astype(np.int64)
    ok = (start >= 0) & (stop <= envelope.shape[0])
    out = np.full((len(start), envelope.shape[1]), np.nan)
    out[ok] = (cs[stop[ok]] - cs[start[ok]]) / (stop[ok] - start[ok])[:, None]
    return out, ok

# ---------------- audio / mel ----------------
def mel_verwoert(audio16k_int16range, n_mel=MEL_BINS, sr=16000, win=WIN_SEC, shift=0.01):
    """extractMelSpecs of Verwoert with n_mel filters (their code uses 23). audio: float array scaled to the int16 range."""
    n_win = int(np.floor((audio16k_int16range.shape[0] - win * sr) / (shift * sr)))
    w = scipy.signal.windows.hann(int(np.floor(win * sr + 1)), sym=True)[:-1]        # == scipy.hanning(...)[:-1]
    spec = np.zeros((n_win, int(np.floor(win * sr / 2 + 1))), dtype='complex')
    for i in range(n_win):
        s = int(np.floor((i * shift) * sr)); e = int(np.floor(s + win * sr))
        spec[i, :] = np.fft.rfft(w * audio16k_int16range[s:e])
    mfb = MelFilterBank(spec.shape[1], n_mel, sr)
    return mfb.toLogMels(np.abs(spec)).astype('float')

def mel_hifigan_speechbrain(audio22k_float):
    """Exactly the call prescribed by the model card of speechbrain/tts-hifigan-ljspeech. Needs: pip install speechbrain torch."""
    import torch
    try:                                                                             # newer SpeechBrain spells it mel_spectrogram,
        from speechbrain.lobes.models.FastSpeech2 import mel_spectrogram as mel_spectogram   # older ones (model card) mel_spectogram
    except ImportError:
        from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
    spec, _ = mel_spectogram(sample_rate=22050, hop_length=256, win_length=None, n_mels=MEL_BINS, n_fft=1024, f_min=0.0,
                             f_max=8000.0, power=1, normalized=False, min_max_energy_norm=True, norm="slaney",
                             mel_scale="slaney", compression=True, audio=torch.from_numpy(audio22k_float.astype(np.float32)))
    return spec.numpy().T                                                            # (T, 80), frame i centred at 256 i samples

# ---------------- one recording ----------------
def process_recording(eeg, eeg_sr, audio, audio_sr, labels, mode, mel_fn_hifigan=mel_hifigan_speechbrain):
    """eeg (N, C); audio (M,) at audio_sr (48 kHz in this dataset); labels (N,) at eeg_sr or None."""
    audio = np.asarray(audio, dtype=np.float64)
    if mode == "verwoert":
        a = scipy.signal.decimate(audio, int(audio_sr / 16000)); a_sr = 16000      # as in the original script
        a = np.int16(a / np.max(np.abs(a)) * 32767).astype(np.float64)               # as in the original script (int16 range)
        spec = mel_verwoert(a)
        starts = 0.01 * np.arange(spec.shape[0])                                    # frame i: [0.01 i, 0.01 i + 0.05]
        centers = starts + WIN_SEC / 2
        wav = (a / 32767.0).astype(np.float32)
    elif mode == "hifigan":
        assert audio_sr == 48000, "this script assumes the 48 kHz audio of the dataset; adjust resample_poly otherwise"
        a = scipy.signal.resample_poly(audio, 147, 320)                              # 48000 * 147/320 = 22050 exactly
        a = a / np.max(np.abs(a)); a_sr = 22050
        spec = mel_fn_hifigan(a)
        centers = 256.0 * np.arange(spec.shape[0]) / 22050.0
        starts = centers - WIN_SEC / 2
        wav = a.astype(np.float32)
    else:
        raise ValueError(mode)
    env = hg_envelope(np.asarray(eeg, dtype=np.float64), eeg_sr)
    feat, ok = window_means(env, eeg_sr, starts)
    keep = np.where(ok)[0]
    i0, i1 = keep[0], keep[-1] + 1
    assert ok[i0:i1].all(), "valid frames must be contiguous"
    feat, spec, centers = feat[i0:i1].astype(np.float32), spec[i0:i1].astype(np.float32), centers[i0:i1]
    lab = None
    if labels is not None:
        idx = np.clip(np.floor(centers * eeg_sr).astype(int), 0, len(labels) - 1)
        lab = np.asarray(labels)[idx]
    assert feat.shape[0] == spec.shape[0] and np.isfinite(feat).all() and np.isfinite(spec).all()
    return dict(feat=feat, spec=spec, frame_times=centers, labels=lab, wav=wav, wav_sr=a_sr, first_frame=int(i0))

def lag_diagnostic(feat, spec, frame_step, max_lag_s=1.0):
    """Correlation between the mean z-scored HG envelope and the mean log-mel over lags; prints the lag with the peak.
    A positive lag means the audio follows the neural signal (expected for speech production, typically 0.05-0.3 s)."""
    x = feat.mean(axis=1); y = spec.mean(axis=1)
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    L = int(max_lag_s / frame_step); lags = np.arange(-L, L + 1)
    cc = [np.mean(x[max(0, -l):len(x) - max(0, l)] * y[max(0, l):len(y) - max(0, -l)]) for l in lags]
    k = int(np.argmax(cc)); return lags[k] * frame_step, cc[k]

# ---------------- main (real data) ----------------
def main(mode):
    from pynwb import NWBHDF5IO
    import pandas as pd
    root, out = os.environ["SEEG_DATA_ROOT"], os.environ["SEEG_OUT_DIR"]
    os.makedirs(out, exist_ok=True)
    for p in range(1, 11):
        sub = f"sub-{p:02d}"
        io = NWBHDF5IO(os.path.join(root, sub, "ieeg", f"{sub}_task-wordProduction_ieeg.nwb"), 'r')
        nwb = io.read()
        eeg = nwb.acquisition['iEEG'].data[:]; audio = nwb.acquisition['Audio'].data[:]
        words = np.array(nwb.acquisition['Stimulus'].data[:], dtype=str); io.close()
        eeg_sr, audio_sr = 1024, 48000                                                # as in the original script
        ch = pd.read_csv(os.path.join(root, sub, "ieeg", f"{sub}_task-wordProduction_channels.tsv"), sep='\t')['name'].to_numpy().astype(str)
        assert len(ch) == eeg.shape[1], f"{sub}: {len(ch)} channel names vs {eeg.shape[1]} columns"
        assert abs(len(eeg) / eeg_sr - len(audio) / audio_sr) < 1.0, f"{sub}: eeg and audio durations differ"
        r = process_recording(eeg, eeg_sr, audio, audio_sr, words, mode)
        lag, c = lag_diagnostic(r['feat'], r['spec'], float(np.median(np.diff(r['frame_times']))))
        print(f"{sub}: frames {r['feat'].shape[0]}, channels {r['feat'].shape[1]}, mel {r['spec'].shape}; "
              f"peak HG-audio lag {lag * 1000:+.0f} ms (corr {c:.2f})")
        np.save(os.path.join(out, f"{sub}_feat.npy"), r['feat']); np.save(os.path.join(out, f"{sub}_spec.npy"), r['spec'])
        np.save(os.path.join(out, f"{sub}_labels.npy"), r['labels']); np.save(os.path.join(out, f"{sub}_frame_times.npy"), r['frame_times'])
        np.save(os.path.join(out, f"{sub}_channels.npy"), ch)
        scipy.io.wavfile.write(os.path.join(out, f"{sub}_audio_{r['wav_sr']}.wav"), r['wav_sr'], r['wav'])
        json.dump(dict(mode=mode, eeg_sr=eeg_sr, audio_sr=audio_sr, win_sec=WIN_SEC, n_mel=MEL_BINS, first_frame=r['first_frame']),
                  open(os.path.join(out, f"{sub}_meta.json"), "w"))

if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_extraction.py")).read())
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "hifigan")
