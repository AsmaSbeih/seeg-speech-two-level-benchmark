"""
speech_per_word.py -- speech duration per word trial (Results, level 2; Supplementary Table S12).

A word trial runs from its prompt onset to the next prompt onset (the end of the recording for the last word), as in
inword_stageB.py. Its speech frames are the frames of the Otsu speech mask (template_floor_continuous.speech_mask) inside it.

Usage (SEEG_OUT_DIR = the features folder with sub-XX_spec.npy and sub-XX_labels.npy):
    python speech_per_word.py
Prints the macros NXWordsAllSpeechMean/Sd/Min/Max (all words), NXSpeechSMin/Max (participant means, s per word) and
NXSpeechPctMin/Max (share of speech frames in the recording, %), and a per-participant table (Table S12), and writes
<SEEG_OUT_DIR>/speech_per_word.csv (one row per word).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_floor_continuous import speech_mask
from inword_stageB import word_trials, FPS, SUBS

OUT = os.environ["SEEG_OUT_DIR"]
rows, per = [], []
for sub in SUBS:
    spec = np.load(os.path.join(OUT, f"{sub}_spec.npy")).astype(np.float32)
    labels = np.load(os.path.join(OUT, f"{sub}_labels.npy"), allow_pickle=True)
    sp = speech_mask(spec)
    tr = word_trials(labels)
    secs = [sp[a:b].sum() / FPS for _, a, b, _ in tr]
    rows += [dict(participant=sub, trial_index=i, word=w, speech_s=s) for (i, a, b, w), s in zip(tr, secs)]
    per.append(dict(participant=sub, duration_s=len(spec) / FPS, trials=len(tr),
                    speech_s_per_word=float(np.mean(secs)), speech_pct=100 * sp.mean()))
W, P = pd.DataFrame(rows), pd.DataFrame(per)
W.to_csv(os.path.join(OUT, "speech_per_word.csv"), index=False)
print(P.round(3).to_string(index=False), "\n")
print(f"NXWordsAllSpeechMean,{W.speech_s.mean():.3f}   ({len(W)} words)")
print(f"NXWordsAllSpeechSd,{W.speech_s.std(ddof=1):.3f}")
print(f"NXWordsAllSpeechMin,{W.speech_s.min():.2f}")
print(f"NXWordsAllSpeechMax,{W.speech_s.max():.2f}")
print(f"NXSpeechSMin,{P.speech_s_per_word.min():.2f}")
print(f"NXSpeechSMax,{P.speech_s_per_word.max():.2f}")
print(f"NXSpeechPctMin,{P.speech_pct.min():.0f}")
print(f"NXSpeechPctMax,{P.speech_pct.max():.0f}")
