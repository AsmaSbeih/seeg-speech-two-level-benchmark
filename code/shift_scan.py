"""
shift_scan.py -- is the "shifted control" PCC (~0.12-0.39) caused by the task rhythm (trial structure)?

Idea: the task repeats every ~3 s (word on screen ~2 s + fixation). A model fed MISALIGNED neural data can still score
>0 if the neural signal contains the task rhythm (visual/fixation responses) and the speech follows the same rhythm.
If so, the control PCC OSCILLATES with the shift (high when the shift is a whole number of trials, low/negative at half a
trial). If there is no rhythm effect, it stays ~0 for every shift.

Usage (same terminal, SEEG_OUT_DIR set; stageB_verwoert_linear.py in the same folder):
    python shift_scan.py              # sub-02, sub-06, sub-10, 12 shifts spanning ~3 s
Writes shift_scan.csv and shift_scan.png into SEEG_OUT_DIR.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageB_verwoert_linear import stack, run, OUT

SUBS = ["sub-02", "sub-06", "sub-10"]
BASE, STEP_F, N = 12000, 22, 12                    # 22 frames = 0.255 s; 12 steps = 3.06 s ~ one trial period
FRAME_S = 256 / 22050

if __name__ == "__main__":
    rows = []
    for sub in SUBS:
        feat = np.load(os.path.join(OUT, f"{sub}_feat.npy")); spec = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
        X, idx = stack(feat); Y = spec[idx]
        for k in range(N):
            s = BASE + k * STEP_F
            r = run(np.roll(X, s, axis=0), Y, 0).mean()
            rows.append(dict(subject=sub, shift_frames=s, extra_ms=round(k * STEP_F * FRAME_S * 1000), pcc=r))
            print(f"{sub}  shift {s} (+{k * STEP_F * FRAME_S:.2f} s): control PCC {r:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "shift_scan.csv"), index=False)
    piv = df.pivot(index="extra_ms", columns="subject", values="pcc")
    print("\n", piv.round(3).to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ax = piv.plot(marker="o", figsize=(7, 4)); ax.axhline(0, color="k", lw=.5)
        ax.set_xlabel("extra shift beyond 12000 frames (ms)"); ax.set_ylabel("control PCC")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "shift_scan.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)
