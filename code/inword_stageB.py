"""
inword_stageB.py -- word-level ("in-word") results for the per-participant models (10 contiguous folds).

For every word trial and every system: the in-speech PCC computed on THAT WORD's speech frames only
(Pearson correlation per mel bin between prediction and original, averaged over the 80 bins).

Definitions (identical to the rest of the paper):
  * word trial  : starts at a word onset (label change INTO a word in sub-XX_labels.npy, same rule as
                  template_floor_continuous.phase_since_onset) and lasts until the next word onset
                  (the last trial: until the end of the recording). trial_index is 0-based within the participant.
  * speech frames: the SAME Otsu mask as the in-speech PCC (template_floor_continuous.speech_mask on the mean log-mel of the
                  whole original spectrogram), restricted to the frames of the trial. The same frames are used for every system.
  * kept words  : every trial with >= 1 speech frame whose speech frames ALL lie among the predicted test frames of one fold
                  (so they are entirely inside that fold's test block). Words with speech frames in two folds, or in the
                  tail of a block that does not fill a 256-frame window, are not kept. inspeech_pcc is NaN if n_speech_frames < 3.
  * systems     : Linear, NeuroIncept, ConvED-SR, CNN-BiLSTM (=baseline), MR-CNN-BiLSTM (=hybrid), timing template.
                  The template is read from the '_T' array saved with every run (train-fold mean per time-since-onset bin).
  * seeds       : seed 42 from <SEEG_OUT_DIR>/stageB/<model>_<sub>_fold<k>_preds.npz ; seeds 123/456 from
                  <SEEG_OUT_DIR>/stageB_seeds_preds/<model>_<sub>_fold<k>_seed<s>_preds.npz IF those files exist
                  (stageB_extra_seeds.py does not save predictions; see stageB_extra_seeds_savepreds.py).
                  Linear and the template are deterministic: they are written once, with seed 42.

Check built in: for each participant-fold and system the script also recomputes the usual in-speech PCC (all speech frames of
the fold) from the same predictions and compares it with pcc_in_speech in the saved <...>_fold<k>.json. If they do not match
(tolerance 1e-4) it stops: that would mean the frames/mask are not the ones of the paper.

Usage (SEEG_OUT_DIR set; run from the code folder):
    python inword_stageB.py                    # all participants, all folds, all seeds found
    python inword_stageB.py sub-02 sub-06      # only these participants
Output: <SEEG_OUT_DIR>/inword_stageB_long.csv  with columns
    participant, fold, seed, trial_index, word, onset_s, n_speech_frames, system, inspeech_pcc
(fold is 0-indexed as in the code: fold k tests block k; paper fold = k+1.)
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_floor_continuous import speech_mask, phase_since_onset, NONWORD

OUT = os.environ["SEEG_OUT_DIR"]                 # set SEEG_OUT_DIR before running
FPS = 22050 / 256                                   # 86.13 frames per second (HiFi-GAN grid)
SYSTEMS = [("linear", "Linear"), ("neuroincept", "NeuroIncept"), ("convedsr", "ConvED-SR"),
           ("baseline", "CNN-BiLSTM"), ("hybrid", "MR-CNN-BiLSTM")]
SUBS = [f"sub-{i:02d}" for i in range(1, 11)]
NFOLD = 10


def pcc_bins(Y, P):                                  # same as stageA_pooled.pcc_bins
    Y = Y - Y.mean(0); P = P - P.mean(0)
    return (Y * P).sum(0) / (np.sqrt((Y ** 2).sum(0) * (P ** 2).sum(0)) + 1e-12)


def pcc_word(Y, P):
    """mean over the 80 mel bins of the Pearson correlation on these frames; NaN if fewer than 3 frames"""
    if len(Y) < 3:
        return float("nan")
    Yc = Y - Y.mean(0); Pc = P - P.mean(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    r = np.where(den > 1e-12, (Yc * Pc).sum(0) / np.maximum(den, 1e-12), np.nan)
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


def word_trials(labels):
    lab = np.asarray(labels).astype(str)
    is_word = np.array([l not in NONWORD for l in lab])
    onset = np.where(is_word & np.concatenate([[True], lab[1:] != lab[:-1]]))[0]
    assert len(onset) == phase_since_onset(labels)[1]
    ends = np.concatenate([onset[1:], [len(lab)]])
    return [(i, int(a), int(b), lab[a]) for i, (a, b) in enumerate(zip(onset, ends))]


def files_for(model, sub, k):
    out = []
    p = os.path.join(OUT, "stageB", f"{model}_{sub}_fold{k}_preds.npz")
    if os.path.exists(p):
        out.append((42, p, os.path.join(OUT, "stageB", f"{model}_{sub}_fold{k}.json")))
    if model != "linear":
        for s in (123, 456):
            q = os.path.join(OUT, "stageB_seeds_preds", f"{model}_{sub}_fold{k}_seed{s}_preds.npz")
            if os.path.exists(q):
                out.append((s, q, os.path.join(OUT, "stageB_seeds_preds", f"{model}_{sub}_fold{k}_seed{s}.json")))
    return out


def main(subs):
    rows, n_missing, n_checked = [], 0, 0
    for sub in subs:
        spec = np.load(os.path.join(OUT, f"{sub}_spec.npy")).astype(np.float32)
        labels = np.load(os.path.join(OUT, f"{sub}_labels.npy"), allow_pickle=True)
        sp = speech_mask(spec)
        trials = word_trials(labels)
        for k in range(NFOLD):
            tmpl_done = False
            for model, name in SYSTEMS:
                for seed, pf, jf in files_for(model, sub, k):
                    z = np.load(pf); idx = z[sub + "_idx"]; P = z[sub + "_P"]; T = z[sub + "_T"]
                    row_of = np.full(len(spec), -1, dtype=np.int64); row_of[idx] = np.arange(len(idx))
                    # ---- check against the saved in-speech PCC of the paper
                    spi = sp[idx]
                    chk = float(pcc_bins(spec[idx][spi], P[spi]).mean()) if spi.sum() > 100 else float("nan")
                    ref = json.load(open(jf))["per_subject"][sub]["pcc_in_speech"] if os.path.exists(jf) else None
                    if ref is not None and np.isfinite(ref):
                        assert abs(chk - ref) < 1e-4, f"{sub} fold {k} {name} seed {seed}: recomputed in-speech PCC {chk:.5f} != saved {ref:.5f}"
                        n_checked += 1
                    systems = [(name, P)]
                    if not tmpl_done and seed == 42:
                        systems.append(("Timing template", T)); tmpl_done = True
                    for i, a, b, w in trials:
                        F = np.where(sp[a:b])[0] + a
                        if len(F) == 0:
                            continue
                        r = row_of[F]
                        if (r < 0).any():
                            continue                                  # speech frames not all inside this fold's predicted test frames
                        for sname, PP in systems:
                            rows.append(dict(participant=sub, fold=k, seed=seed, trial_index=i, word=w, onset_s=round(a / FPS, 4),
                                             n_speech_frames=len(F), system=sname,
                                             inspeech_pcc=pcc_word(spec[F], PP[r])))
        print(f"{sub}: done ({len(trials)} word trials)")
    df = pd.DataFrame(rows, columns=["participant", "fold", "seed", "trial_index", "word", "onset_s", "n_speech_frames", "system", "inspeech_pcc"])
    out = os.path.join(OUT, "inword_stageB_long.csv"); df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {out} | in-speech PCC re-computation matched the saved values in {n_checked} run files")
    d42 = df[df.seed == 42]
    print("\nwords kept per participant (seed 42, one system):")
    print(d42[d42.system == "MR-CNN-BiLSTM"].groupby("participant").size().to_string())
    print("\nmean in-word PCC over words (seed 42), all words / words with >= 20 speech frames:")
    t = pd.DataFrame({"all": d42.groupby("system").inspeech_pcc.mean(),
                      ">=20 frames": d42[d42.n_speech_frames >= 20].groupby("system").inspeech_pcc.mean(),
                      "n words": d42.groupby("system").size()})
    print(t.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or SUBS)
