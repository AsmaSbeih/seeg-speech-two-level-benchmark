"""Reconstruction figure in the layout of ConvED-SR Fig. 3: log-mel (left) and waveform (right), original (top) and reconstructed (bottom).
One call = one figure (call it once for Stage A and once for Stage B). Waveform of the reconstruction = pretrained HiFi-GAN decode of the predicted log-mel
(the same vocoder as STOI/MCD); original waveform = the recorded audio of the same frames. Both waveforms are peak-normalised (the vocoder gain is arbitrary).
Same participant and same 15 s segment in Stage A and Stage B: Stage B fold 2 tests block index 2, which is the first Stage A test block."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
HOP, SR = 256, 22050


def get_decoder(OUT):
    """HiFi-GAN from the LOCAL checkpoint folder <OUT>/hifigan_ckpt (no download: avoids the huggingface_hub 'use_auth_token' error)."""
    import torch
    from speechbrain.inference.vocoders import HIFIGAN
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = os.path.join(OUT, "hifigan_ckpt"); kw = {}
    try:
        from speechbrain.utils.fetching import LocalStrategy
        kw["local_strategy"] = LocalStrategy.COPY
    except Exception:
        pass
    hifi = HIFIGAN.from_hparams(source=ck, savedir=ck, run_opts={"device": dev}, **kw)

    def decode(mel_t80):                                          # (T, 80) raw log-mel -> waveform
        with torch.no_grad():
            m = torch.tensor(mel_t80.T, dtype=torch.float32)[None].to(dev)
            return hifi.decode_batch(m).squeeze().cpu().numpy()
    return decode


def best_subject(stageB_dir, model="hybrid", key="pcc"):
    """participant with the highest mean `key` over the 10 folds of `model` (Stage B JSON files)"""
    v = {}
    for f in glob.glob(os.path.join(stageB_dir, f"{model}_sub-*_fold*.json")):
        j = json.load(open(f)); v.setdefault(j["subject"], []).append(j["per_subject"][j["subject"]][key])
    m = {s: float(np.mean(a)) for s, a in v.items()}
    return max(m, key=m.get), m


def make(sub, npz, spec, x, decode, out, seconds=15.0, block_label="3", title=None, labels=None):
    """npz: preds file (<sub>_idx, <sub>_P); spec: (T,80) original log-mel of the whole recording; x: original audio (22.05 kHz) of the whole recording;
    decode: function (T,80) raw log-mel -> waveform (stageA_stoi.get_vocoder())"""
    z = np.load(npz, allow_pickle=True); idx, P = z[sub + "_idx"], z[sub + "_P"]
    brk = np.where(np.diff(idx) != 1)[0]
    n_run = brk[0] + 1 if len(brk) else len(idx)                    # first contiguous test block
    n = min(int(seconds * SR / HOP), n_run); fr = idx[:n]
    Y, Pm = spec[fr], P[:n]
    ref = x[int(fr[0]) * HOP: (int(fr[-1]) + 1) * HOP]; rec = decode(Pm); m = min(len(ref), len(rec)); ref, rec = ref[:m], rec[:m]
    ref = ref / (np.abs(ref).max() + 1e-9); rec = rec / (np.abs(rec).max() + 1e-9)
    vmin, vmax = np.percentile(Y, 1), np.percentile(Y, 99.5); dur = n * HOP / SR; tw = np.arange(m) / SR
    fig, ax = plt.subplots(2, 2, figsize=(7.16, 3.7), gridspec_kw=dict(width_ratios=[1.05, 1], hspace=0.12, wspace=0.28))
    for r, (M, w, lab) in enumerate(((Y, ref, "Original"), (Pm, rec, "Reconstructed"))):
        a = ax[r, 0]
        a.imshow(M.T, origin="lower", aspect="auto", extent=[0, dur, 0, 80], vmin=vmin, vmax=vmax, cmap="viridis")
        a.set_ylabel("Mel frequency bins", fontsize=8); a.tick_params(labelsize=7)
        b = ax[r, 1]
        b.plot(tw, w, lw=0.4, color="#4F8FC0"); b.set_xlim(0, dur); b.set_ylim(-1.05, 1.05)
        b.set_ylabel("Amplitude (normalised)", fontsize=8); b.tick_params(labelsize=7)
        b.yaxis.set_label_position("left"); b.text(1.02, 0.5, lab, transform=b.transAxes, rotation=90, va="center", ha="left", fontsize=8)
        if r == 0: a.set_xticklabels([]); b.set_xticklabels([])
    for a in (ax[1, 0], ax[1, 1]): a.set_xlabel("Time (s)", fontsize=8)
    ax[0, 0].set_title("Log-mel spectrogram", fontsize=8); ax[0, 1].set_title("Audio waveform", fontsize=8)
    if labels is not None:                                           # prompted word written above the original spectrogram (centre of its labelled span)
        try:
            from template_floor_continuous import NONWORD
        except Exception:
            NONWORD = {"", "nan", "None", "silence", "sil", "pause"}
        lab = np.asarray(labels).astype(str)[fr]; i = 0
        while i < n:
            j = i
            while j + 1 < n and lab[j + 1] == lab[i]: j += 1
            if lab[i] not in NONWORD:
                t0, t1 = i * HOP / SR, (j + 1) * HOP / SR
                ax[0, 0].plot([t0, t1], [1.015, 1.015], transform=ax[0, 0].get_xaxis_transform(), color="#444", lw=0.8, clip_on=False)
                ax[0, 0].text((t0 + t1) / 2, 1.03, lab[i], transform=ax[0, 0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7, style="italic")
            i = j + 1
        ax[0, 0].set_title("Log-mel spectrogram", fontsize=8, pad=16); ax[0, 1].set_title("Audio waveform", fontsize=8, pad=16)
    if title: fig.suptitle(title, fontsize=8, y=1.0)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    print("saved", out, "|", sub, "| seconds", round(dur, 1), "| frames", n)


def make_both(sub, npzA, npzB, spec, x, decode, out, seconds=15.0, title=None, labels=None):
    """ONE figure, same 15 s: row 1 original, row 2 (A) pooled multi-subject Hybrid, row 3 (B) subject-dependent Hybrid.
    Left: log-mel; right: waveform (HiFi-GAN decode of the prediction; peak-normalised). Words above the original spectrogram = prompted words."""
    zA, zB = np.load(npzA, allow_pickle=True), np.load(npzB, allow_pickle=True)
    iA, PA, iB, PB = zA[sub + "_idx"], zA[sub + "_P"], zB[sub + "_idx"], zB[sub + "_P"]
    common = np.intersect1d(iA, iB); brk = np.where(np.diff(common) != 1)[0]
    run = common[: (brk[0] + 1 if len(brk) else len(common))]
    n = min(int(seconds * SR / HOP), len(run)); fr = run[:n]
    Y = spec[fr]; A = PA[np.searchsorted(iA, fr)]; B = PB[np.searchsorted(iB, fr)]
    ref = x[int(fr[0]) * HOP: (int(fr[-1]) + 1) * HOP]
    waves = [ref, decode(A), decode(B)]; m = min(len(w) for w in waves)
    waves = [w[:m] / (np.abs(w[:m]).max() + 1e-9) for w in waves]
    vmin, vmax = np.percentile(Y, 1), np.percentile(Y, 99.5); dur = n * HOP / SR; tw = np.arange(m) / SR
    rows = [(Y, waves[0], "Original"), (A, waves[1], "(A) Pooled"), (B, waves[2], "(B) Per-\nparticipant")]
    fig, ax = plt.subplots(3, 2, figsize=(7.16, 5.3), gridspec_kw=dict(width_ratios=[1.05, 1], hspace=0.14, wspace=0.28))
    for r, (M, w, lab) in enumerate(rows):
        a = ax[r, 0]; a.imshow(M.T, origin="lower", aspect="auto", extent=[0, dur, 0, 80], vmin=vmin, vmax=vmax, cmap="viridis")
        a.set_ylabel("Mel frequency bins", fontsize=8); a.tick_params(labelsize=7)
        b = ax[r, 1]; b.plot(tw, w, lw=0.4, color="#4F8FC0"); b.set_xlim(0, dur); b.set_ylim(-1.05, 1.05)
        b.set_ylabel("Amplitude (normalised)", fontsize=8); b.tick_params(labelsize=7)
        b.text(1.02, 0.5, lab, transform=b.transAxes, rotation=90, va="center", ha="left", fontsize=7.5, linespacing=1.0)
        if r < 2: a.set_xticklabels([]); b.set_xticklabels([])
    for a in (ax[2, 0], ax[2, 1]): a.set_xlabel("Time (s)", fontsize=8)
    ax[0, 0].set_title("Log-mel spectrogram", fontsize=8, pad=16); ax[0, 1].set_title("Audio waveform", fontsize=8, pad=16)
    if labels is not None:
        try:
            from template_floor_continuous import NONWORD
        except Exception:
            NONWORD = {"", "+", "nan", "None"}
        lab = np.asarray(labels).astype(str)[fr]; i = 0
        while i < n:
            j = i
            while j + 1 < n and lab[j + 1] == lab[i]: j += 1
            if lab[i] not in NONWORD:
                t0, t1 = i * HOP / SR, (j + 1) * HOP / SR
                ax[0, 0].plot([t0, t1], [1.015, 1.015], transform=ax[0, 0].get_xaxis_transform(), color="#444", lw=0.8, clip_on=False)
                ax[0, 0].text((t0 + t1) / 2, 1.03, lab[i], transform=ax[0, 0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7, style="italic")
            i = j + 1
    if title: fig.suptitle(title, fontsize=8, y=1.0)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    print("saved", out, "|", sub, "| seconds", round(dur, 1), "| frames", n)
