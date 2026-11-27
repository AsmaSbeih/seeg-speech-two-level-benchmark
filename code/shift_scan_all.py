"""
shift_scan_all.py -- the shift check of shift_scan.py (Verwoert-style linear decoder, inputs circularly shifted by 12000 frames + 0..2.8 s, 12 shifts)
for ALL 10 participants (the manuscript's preliminary check used participants 2, 6 and 10). Supplementary material.

Idea: the task repeats every ~3 s, so a misaligned neural signal can still score above zero; a shifted signal is then not a valid null for this task.

Usage (SEEG_OUT_DIR set; stageB_verwoert_linear.py in the same folder):
    python shift_scan_all.py
Writes shift_scan_all.csv (all participants) and prints a per-participant summary. Existing shift_scan.csv is not touched.
Time: about 3x that of shift_scan.py.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageB_verwoert_linear import stack, run, OUT

SUBS = [f"sub-{i:02d}" for i in range(1, 11)]
BASE, STEP_F, N = 12000, 22, 12                    # 22 frames = 0.255 s; 12 steps = 3.06 s ~ one trial period
FRAME_S = 256 / 22050

if __name__ == "__main__":
    f_out = os.path.join(OUT, "shift_scan_all.csv")
    done = pd.read_csv(f_out) if os.path.exists(f_out) else pd.DataFrame(columns=["subject", "shift_frames", "extra_ms", "pcc"])
    rows = done.to_dict("records")
    for sub in SUBS:
        if (done.subject == sub).sum() == N:
            print(sub, "already done"); continue
        rows = [r for r in rows if r["subject"] != sub]
        feat = np.load(os.path.join(OUT, f"{sub}_feat.npy")); spec = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
        X, idx = stack(feat); Y = spec[idx]
        for k in range(N):
            s = BASE + k * STEP_F
            r = run(np.roll(X, s, axis=0), Y, 0).mean()
            rows.append(dict(subject=sub, shift_frames=s, extra_ms=round(k * STEP_F * FRAME_S * 1000), pcc=float(r)))
            print(f"{sub}  shift {s} (+{k * STEP_F * FRAME_S:.2f} s): control PCC {r:+.3f}", flush=True)
        pd.DataFrame(rows).to_csv(f_out, index=False)                 # saved after every participant (resumable)
    df = pd.DataFrame(rows)
    s = df.groupby("subject")["pcc"].agg(["min", "max", "mean", lambda x: int((x > 0).sum())])
    s.columns = ["min", "max", "mean", "n_positive_of_12"]
    print("\nper participant:\n", s.round(3).to_string())
    print(f"\nall participants: {int((df.pcc > 0).sum())} of {len(df)} shifts above zero; range {df.pcc.min():+.3f} to {df.pcc.max():+.3f}; mean {df.pcc.mean():+.3f}")
