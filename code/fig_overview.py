"""New Fig. 1: original log-mel, no-neural-input template and Hybrid decoder over four consecutive trials (participant 6, test block 3).
Inputs: <OUT>/sub-06_spec.npy, <OUT>/sub-06_labels.npy, <OUT>/stageB/hybrid_sub-06_fold2_preds.npz (keys sub-06_idx, sub-06_P, sub-06_T).
Usage: python fig_overview.py OUT_DIR OUTPUT_PREFIX [participant] [fold]      e.g.  python fig_overview.py C:\\seeg_v2\\features_hifigan C:\\seeg_v2\\fig_overview sub-06 2
(fold = code fold, 0-based; fold 2 = paper fold 3 = test block 3). Needs only files that already exist; trains nothing."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
OUT, PREFIX = sys.argv[1], sys.argv[2]
SUB = sys.argv[3] if len(sys.argv) > 3 else "sub-06"; FOLD = int(sys.argv[4]) if len(sys.argv) > 4 else 2; HOP, SR = 256, 22050
spec = np.load(f"{OUT}/{SUB}_spec.npy"); lab = np.load(f"{OUT}/{SUB}_labels.npy", allow_pickle=True).astype(str)
z = np.load(f"{OUT}/stageB/hybrid_{SUB}_fold{FOLD}_preds.npz", allow_pickle=True)
idx, P, T = z[f"{SUB}_idx"], z[f"{SUB}_P"], z[f"{SUB}_T"]
Y = spec[idx]; l = lab[idx]
on = [i for i in range(1, len(l)) if l[i] != "" and l[i] != l[i - 1]]           # word onsets inside the test block
a = on[0] - 12; n_tr = 4; b = on[n_tr] - 12                                   # four consecutive trials (word + pause)
Ys, Ps, Ts, ls = Y[a:b], P[a:b], T[a:b], l[a:b]; dur = (b - a) * HOP / SR
vmin, vmax = np.percentile(Ys, 1), np.percentile(Ys, 99.5)
rows = [("Original speech", Ys, None), ("Template\n(no neural input)", Ts, None), ("Decoded from sEEG\n(MR-CNN-BiLSTM)", Ps, None)]
fig, ax = plt.subplots(3, 1, figsize=(7.16, 3.6), sharex=True, gridspec_kw=dict(hspace=0.08))
for k, (name, M, _) in enumerate(rows):
    ax[k].imshow(M.T, origin="lower", aspect="auto", extent=[0, dur, 0, 80], vmin=vmin, vmax=vmax, cmap="viridis")
    ax[k].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center", labelpad=6); ax[k].set_yticks([0, 80]); ax[k].tick_params(labelsize=7)
ax[2].set_xlabel("Time (s)", fontsize=8)
t = 0
for k in range(n_tr):                                                          # trial labels and word spans above the original
    i0 = on[k] - on[0] + 12; i1 = i0
    while i1 < len(ls) and ls[i1] == ls[i0]: i1 += 1
    t0, t1 = i0 * HOP / SR, i1 * HOP / SR
    ax[0].plot([t0, t1], [1.04, 1.04], transform=ax[0].get_xaxis_transform(), color="#333", lw=1.2, clip_on=False)
    ax[0].text((t0 + t1) / 2, 1.07, ls[i0], transform=ax[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7, style="italic")
fig.text(0.015, 0.5, "Mel bins", rotation=90, va="center", fontsize=8)
fig.subplots_adjust(left=0.2, right=0.985, top=0.9, bottom=0.14)
for e in ("png", "pdf"): fig.savefig(f"{PREFIX}.{e}", dpi=300, bbox_inches=None)
print("done", dur, "s")
