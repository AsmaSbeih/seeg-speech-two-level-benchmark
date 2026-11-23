# ===== Cell: constant-spectrum control for Stage A (STOI / eSTOI), self-contained, resumable =====
import os, sys, glob, json, time
import numpy as np
OUT = os.environ["SEEG_OUT_DIR"]                 # set SEEG_OUT_DIR before running (folder with the extracted features)
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # code folder
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import stageA_stoi as S
from pystoi import stoi

def get_decoder_local(OUT):
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
    def decode(mel_t80):
        with torch.no_grad():
            m = torch.tensor(mel_t80.T, dtype=torch.float32)[None].to(dev)
            return hifi.decode_batch(m).squeeze().cpu().numpy()
    return decode

decode = get_decoder_local(OUT)
QD = os.path.join(OUT, "quality"); os.makedirs(QD, exist_ok=True)
JS = os.path.join(QD, "stageA_constant_control.json")
res = json.load(open(JS)) if os.path.exists(JS) else {}
f_any = S.any_pred_file(); z = np.load(f_any)          # same test frames as the Stage-A STOI of the paper (seed 42)
print("test frames taken from:", os.path.basename(f_any))

for sub in S.ALL_SUBS:
    key = f"A|constant|{sub}|-"
    if key in res: continue
    t0 = time.time()
    spec = np.load(os.path.join(OUT, f"{sub}_spec.npy"))
    sr, x = S.read_wav(glob.glob(os.path.join(OUT, f"{sub}_audio_*.wav"))[0]); assert sr == S.SR
    idx = z[sub + "_idx"]
    test = np.zeros(len(spec), bool); test[idx] = True
    const = spec[~test].mean(0)                          # mean log-mel of ALL non-test frames of this participant (as in Stage B)
    v, e = [], []
    for a, b in S.runs_of(idx):                          # one run = one test block (identical to stageA_stoi.py)
        fr0, fr1 = int(idx[a]), int(idx[b - 1]) + 1
        y = decode(np.tile(const, (b - a, 1)).astype(np.float32))
        ref = x[fr0 * S.HOP: fr1 * S.HOP]; n = min(len(ref), len(y))
        v.append(stoi(ref[:n], y[:n], S.SR, extended=False)); e.append(stoi(ref[:n], y[:n], S.SR, extended=True))
    res[key] = dict(stage="A", system="constant", subject=sub, stoi=float(np.mean(v)), estoi=float(np.mean(e)), blocks=len(v))
    json.dump(res, open(JS, "w"), indent=1)
    print(f"{sub}: constant STOI {np.mean(v):.3f} | eSTOI {np.mean(e):.3f} | {len(v)} blocks | {time.time()-t0:.0f}s", flush=True)

# ---- comparison with the systems already in the paper (Stage A, seed 42)
import pandas as pd
pd.set_option("display.width", 200)
old = pd.DataFrame(json.load(open(os.path.join(OUT, "stageA_stoi", "stoi_results.json"))).values())
new = pd.DataFrame([r for r in res.values()])
allr = pd.concat([old[["subject", "system", "stoi", "estoi"]], new[["subject", "system", "stoi", "estoi"]]])
print("\nSTOI per participant:"); print(allr.pivot(index="subject", columns="system", values="stoi").round(3).to_string())
m = allr.groupby("system")[["stoi", "estoi"]].mean().round(3)
print("\nmean over participants:"); print(m.to_string())
c = m.loc["constant", "stoi"]
print(f"\nconstant STOI = {c:.3f};  systems with mean STOI ABOVE it:", [s for s in m.index if s not in ('constant', 'truth') and m.loc[s, 'stoi'] > c])
print("saved ->", JS)
