"""
stageA_pooled.py -- STAGE A: one model for ALL participants (zero-padding to 127 channels + mask + subject embedding),
trained on the NEW continuous 80-bin features (extract_continuous_80.py, hifigan mode).

Models in this file:  baseline (CNN-BiLSTM), hybrid (CNN-BiLSTM + 5 MR-ResBlocks), linear (Verwoert-style PCA+linear, pooled)
(NeuroIncept and ConvED-SR are added in the next step, with the same split / loss / optimiser / lr grid / seeds.)

Design (identical for every model)
  * each recording is cut into 10 contiguous blocks (~30 s):  test = blocks 3 and 8, validation = blocks 1 and 6, train = the rest
    (1-indexed). Windows never cross block boundaries, so no frame is shared between train / val / test.
  * windows of 256 frames (2.97 s). Train windows: stride 64. Val/test windows: stride 256 (every test frame predicted once).
  * inputs: HG features z-scored per participant (train-block statistics), padded to 127 channels with a mask.
    targets, early stopping on validation MSE (patience 6).
  * learning-rate protocol (same for every neural model): grid {1e-4, 3e-4, 1e-3}, chosen on validation MSE with seed 42,
    then seeds 42 / 123 / 456 at the chosen lr (the seed-42 run is reused).
  * metrics per participant on the test frames (all concatenated, in time order): PCC per mel bin, mean over bins (ConvED-SR
    definition); the same inside speech frames only; the TIME-IN-TRIAL TEMPLATE (no neural input) and the residual PCC.
    Predictions are saved (raw log-mel) for the later HiFi-GAN / STOI step.

Usage (terminal, SEEG_OUT_DIR set; template_floor_continuous.py and stageB_verwoert_linear.py in the same folder):
    python stageA_pooled.py smoke run baseline      # 1 epoch, 2 participants: checks the plumbing and prints the time per epoch
    python stageA_pooled.py run baseline            # lr search + 3 seeds
    python stageA_pooled.py run hybrid
    python stageA_pooled.py run linear
    python stageA_pooled.py run baseline_null       # SAME model, inputs replaced by noise (1 run, lr of the real model): reference for
    python stageA_pooled.py run hybrid_null         # the residual / in-speech metrics (what the model scores without neural input)
    python stageA_pooled.py summary
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_floor_continuous import phase_since_onset, template_predict, speech_mask, MAXPH

OUT = os.environ["SEEG_OUT_DIR"]
SMOKE = sys.argv[1:2] == ["smoke"]
RES = os.path.join(OUT, "stageA_smoke" if SMOKE else "stageA")
os.makedirs(RES, exist_ok=True)

CH, MEL, W = 127, 80, 256
NBLK, TEST_B, VAL_B = 10, (2, 7), (0, 5)
STRIDE_TRAIN, BS, MAX_EP, PATIENCE = 64, 16, 60, 8
WEIGHT_DECAY = 1e-4                                                       # as in the original notebook
LR_FACTOR, LR_PATIENCE = 0.5, 2                                           # ReduceLROnPlateau(factor 0.5, patience 2), as in the original notebook
LR_GRID, SEEDS = [1e-4, 3e-4, 1e-3], [42, 123, 456]
ALL_SUBS = [f"sub-{i:02d}" for i in range(1, 11)]
DEV = "cpu"
NULL = False                                                              # set by the model name "<model>_null": inputs replaced by fresh N(0,1) noise


# ------------------------------------------------------------------ models (copied from Complete_Pipeline.ipynb)
class MRResBlock(nn.Module):
    def __init__(self, ch, kernels=(3, 5, 7), norm='bn', dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(nn.Sequential(
                nn.Conv1d(ch, ch, k, padding=k // 2),
                nn.BatchNorm1d(ch) if norm == 'bn' else nn.GroupNorm(min(32, ch), ch),
                nn.LeakyReLU(0.2)))
        self.merge = nn.Sequential(
            nn.Conv1d(ch * len(kernels), ch, 1),
            nn.BatchNorm1d(ch) if norm == 'bn' else nn.GroupNorm(min(32, ch), ch),
            nn.LeakyReLU(0.2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.merge(torch.cat([b(x) for b in self.branches], dim=1)))


class BaselineModel(nn.Module):
    def __init__(self, n_sub=10, emb=32, hid=256):
        super().__init__()
        self.sub_embed = nn.Embedding(n_sub, emb)
        self.encoder = nn.Sequential(
            nn.Conv1d(CH, 128, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2))
        self.pool = nn.AdaptiveAvgPool1d(W)
        self.lstm = nn.LSTM(256 + emb, hid, 1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid * 2, MEL)

    def forward(self, x, mask, sid):
        z = self.pool(self.encoder(x * mask)).permute(0, 2, 1)
        e = self.sub_embed(sid).unsqueeze(1).expand(-1, z.size(1), -1)
        return self.proj(self.lstm(torch.cat([z, e], dim=-1))[0]).permute(0, 2, 1)


class HybridModel(nn.Module):
    def __init__(self, n_sub=10, emb=32, hid=256, kernels=(3, 5, 7), norm='bn'):
        super().__init__()
        self.sub_embed = nn.Embedding(n_sub, emb)
        self.conv1 = nn.Sequential(nn.Conv1d(CH, 128, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 256, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2))
        self.mr1 = MRResBlock(128, kernels, norm)
        self.mr2 = MRResBlock(256, kernels, norm)
        self.mr3 = MRResBlock(256, kernels, norm)
        self.mr_pre = MRResBlock(256, kernels, norm)
        self.mr_post = MRResBlock(256, kernels, norm)
        self.pool = nn.AdaptiveAvgPool1d(W)
        self.lstm = nn.LSTM(256 + emb, hid, 1, batch_first=True, bidirectional=True)
        self.post_proj = nn.Linear(hid * 2, 256)
        self.final_proj = nn.Linear(256, MEL)

    def forward(self, x, mask, sid):
        z = self.mr1(self.conv1(x * mask))
        z = self.mr2(self.conv2(z))
        z = self.mr_pre(self.mr3(self.conv3(z)))
        z = self.pool(z).permute(0, 2, 1)
        e = self.sub_embed(sid).unsqueeze(1).expand(-1, z.size(1), -1)
        z, _ = self.lstm(torch.cat([z, e], dim=-1))
        z = self.post_proj(z).permute(0, 2, 1)
        z = self.mr_post(z).permute(0, 2, 1)
        return self.final_proj(z).permute(0, 2, 1)


CTORS = {"baseline": BaselineModel, "hybrid": HybridModel}


# ------------------------------------------------------------------ data
def load_all(subs):
    D = {}
    for sub in subs:
        feat = np.load(os.path.join(OUT, f"{sub}_feat.npy")).astype(np.float32)
        spec = np.load(os.path.join(OUT, f"{sub}_spec.npy")).astype(np.float32)
        labels = np.load(os.path.join(OUT, f"{sub}_labels.npy"), allow_pickle=True)
        T = len(feat); e = np.linspace(0, T, NBLK + 1).astype(int)
        D[sub] = dict(sid=ALL_SUBS.index(sub), feat=feat, spec=spec, labels=labels, blocks=[(e[i], e[i + 1]) for i in range(NBLK)])
    # per-participant channel z-score and pooled mel standardisation, both from TRAIN blocks only
    ys = []
    for d in D.values():
        tr = np.concatenate([np.arange(s, e) for b, (s, e) in enumerate(d["blocks"]) if b not in TEST_B + VAL_B])
        mu, sd = d["feat"][tr].mean(0), d["feat"][tr].std(0) + 1e-6
        d["X"] = ((d["feat"] - mu) / sd).astype(np.float32); d["train_idx"] = tr
        ys.append(d["spec"][tr])
    ys = np.concatenate(ys)
    ymu, ysd = ys.mean(0), ys.std(0) + 1e-6
    for d in D.values():
        d["Yn"] = ((d["spec"] - ymu) / ysd).astype(np.float32)
    return D, ymu, ysd


def windows(D, kind):
    out = []
    for sub, d in D.items():
        for b, (s, e) in enumerate(d["blocks"]):
            if kind == "train" and b in TEST_B + VAL_B: continue
            if kind == "val" and b not in VAL_B: continue
            if kind == "test" and b not in TEST_B: continue
            for st in range(s, e - W + 1, STRIDE_TRAIN if kind == "train" else W):
                out.append((sub, st))
    return out


def make_batch(D, wins, idx):
    B = len(idx)
    x = np.zeros((B, CH, W), np.float32); m = np.zeros((B, CH, 1), np.float32)
    y = np.zeros((B, MEL, W), np.float32); sid = np.zeros(B, np.int64)
    for j, i in enumerate(idx):
        sub, st = wins[i]; d = D[sub]; c = d["X"].shape[1]
        x[j, :c] = np.random.randn(c, W) if NULL else d["X"][st:st + W].T; m[j, :c, 0] = 1; y[j] = d["Yn"][st:st + W].T; sid[j] = d["sid"]
    return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y), torch.from_numpy(sid)


@torch.no_grad()
def predict(model, D, wins):
    model.eval(); P = []; np.random.seed(0)
    for i in range(0, len(wins), 32):
        x, m, y, sid = make_batch(D, wins, list(range(i, min(i + 32, len(wins)))))
        P.append(model(x, m, sid).numpy())
    return np.concatenate(P)                                               # (n, 80, W) standardised


def val_mse(model, D, va):
    P = predict(model, D, va)
    Y = np.stack([D[s]["Yn"][st:st + W].T for s, st in va])
    return float(((P - Y) ** 2).mean())


def train_model(name, lr, seed, D, epochs, log):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CTORS[name.replace('_null', '')]().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', LR_FACTOR, patience=LR_PATIENCE)
    tr, va = windows(D, "train"), windows(D, "val")
    best, best_sd, bad = np.inf, None, 0
    for ep in range(epochs):
        t0 = time.time(); model.train(); perm = np.random.permutation(len(tr)); tot = n = 0
        for i in range(0, len(perm), BS):
            ii = perm[i:i + BS]
            if len(ii) < 2: continue
            x, m, y, sid = make_batch(D, tr, ii)
            loss = F.mse_loss(model(x, m, sid), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(ii); n += len(ii)
        v = val_mse(model, D, va); sched.step(v)
        log(f"    [{name} lr={lr:g} seed={seed}] epoch {ep + 1}/{epochs}: train MSE {tot / n:.4f} | val MSE {v:.4f} | {time.time() - t0:.0f}s")
        if v < best - 1e-5:
            best, best_sd, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_sd)
    return model, best


# ------------------------------------------------------------------ metrics
def pcc_bins(Y, P):
    Y = Y - Y.mean(0); P = P - P.mean(0)
    return (Y * P).sum(0) / (np.sqrt((Y ** 2).sum(0) * (P ** 2).sum(0)) + 1e-12)


def evaluate(D, wins_test, P_raw):
    """P_raw: (n_windows, 80, W) predictions in raw log-mel. Returns per-participant metrics + the saved predictions."""
    res, preds = {}, {}
    for sub, d in D.items():
        sel = [i for i, (s, st) in enumerate(wins_test) if s == sub]
        idx = np.concatenate([np.arange(wins_test[i][1], wins_test[i][1] + W) for i in sel])
        P = np.concatenate([P_raw[i].T for i in sel]); Y = d["spec"][idx]
        sp_all = speech_mask(d["spec"]); sp = sp_all[idx]
        ph, _ = phase_since_onset(d["labels"]); tr = d["train_idx"]
        T = template_predict(ph[tr], d["spec"][tr], ph[idx])
        r = dict(pcc=float(pcc_bins(Y, P).mean()), template_pcc=float(pcc_bins(Y, T).mean()),
                 residual_pcc=float(pcc_bins(Y - T, P - T).mean()), mse=float(((Y - P) ** 2).mean()),
                 pcc_in_speech=float(pcc_bins(Y[sp], P[sp]).mean()) if sp.sum() > 100 else float("nan"),
                 template_in_speech=float(pcc_bins(Y[sp], T[sp]).mean()) if sp.sum() > 100 else float("nan"),
                 speech_frac=float(sp.mean()), n_test_frames=int(len(idx)))
        res[sub] = r; preds[sub + "_idx"] = idx; preds[sub + "_P"] = P.astype(np.float32); preds[sub + "_T"] = T.astype(np.float32)
    return res, preds


def save_run(tag, res, preds, extra):
    json.dump(dict(per_subject=res, **extra), open(os.path.join(RES, tag + ".json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RES, tag + "_preds.npz"), **preds)


# ------------------------------------------------------------------ Verwoert-style pooled linear model
def run_linear(D, ymu, ysd, log):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    ORDER, STEP = 4, 5
    offs = np.arange(-ORDER, ORDER + 1) * STEP

    def feats(wins):
        X, S = [], []
        for sub, st in wins:
            d = D[sub]; c = d["X"].shape[1]
            seg = d["X"][st:st + W]
            idx = np.clip(np.arange(W)[:, None] + offs[None, :], 0, W - 1)          # edges replicated inside the window
            z = np.zeros((W, CH, len(offs)), np.float32); z[:, :c, :] = seg[idx].transpose(0, 2, 1)
            X.append(z.reshape(W, -1)); S.append(np.full(W, d["sid"]))
        return np.concatenate(X), np.concatenate(S)

    tr = windows(D, "train")[::W // STRIDE_TRAIN]                          # ~non-overlapping train windows (stride 256)
    te = windows(D, "test")
    Xtr, Str = feats(tr); Ytr = np.concatenate([D[s]["Yn"][st:st + W] for s, st in tr])
    Xte, Ste = feats(te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    pca = PCA(50, random_state=0).fit((Xtr - mu) / sd)
    oh = lambda S: np.eye(10, dtype=np.float32)[S]
    lr = LinearRegression().fit(np.hstack([pca.transform((Xtr - mu) / sd), oh(Str)]), Ytr)
    Pn = lr.predict(np.hstack([pca.transform((Xte - mu) / sd), oh(Ste)]))
    P_raw = (Pn * ysd + ymu).reshape(len(te), W, MEL).transpose(0, 2, 1)
    res, preds = evaluate(D, te, P_raw)
    save_run("linear_seedNA", res, preds, dict(model="linear", train_frames=int(len(Xtr))))
    log(f"linear (pooled, {len(Xtr)} train frames): mean test PCC {np.mean([r['pcc'] for r in res.values()]):.3f}")


# ------------------------------------------------------------------ orchestration
def run_neural(name, D, ymu, ysd, log):
    wins_te = windows(D, "test")
    grid, seeds = ([3e-4], [42]) if SMOKE else (LR_GRID, SEEDS)
    epochs = 1 if SMOKE else MAX_EP
    if NULL and not SMOKE:                                                  # one run only: lr of the real model, seed 42
        import glob, re
        real = glob.glob(os.path.join(RES, name.replace('_null', '') + "_lr*_seed42.json"))
        grid = [float(re.search(r"_lr([0-9.e+-]+)_seed", os.path.basename(real[0])).group(1))] if real else [3e-4]
        seeds = [42]
    cache = {}
    best_lr, best_v = None, np.inf
    for lr in grid:                                                         # lr search on validation MSE, seed 42
        tag = f"{name}_lr{lr:g}_seed42"
        model, v = train_model(name, lr, 42, D, epochs, log)
        cache[lr] = (model, v)
        log(f"  {name}: lr {lr:g} -> best validation MSE {v:.4f}")
        if v < best_v: best_lr, best_v = lr, v
    log(f"  {name}: chosen lr = {best_lr:g}")
    for seed in seeds:
        tag = f"{name}_lr{best_lr:g}_seed{seed}"
        if os.path.exists(os.path.join(RES, tag + ".json")):
            log(f"  {tag}: already done, skipped"); continue
        model, v = cache[best_lr] if seed == 42 else train_model(name, best_lr, seed, D, epochs, log)
        P = predict(model, D, wins_te) * ysd[None, :, None] + ymu[None, :, None]
        res, preds = evaluate(D, wins_te, P)
        save_run(tag, res, preds, dict(model=name, lr=best_lr, seed=seed, val_mse=v))
        log(f"  {tag}: mean test PCC {np.mean([r['pcc'] for r in res.values()]):.3f} | template {np.mean([r['template_pcc'] for r in res.values()]):.3f} "
            f"| residual {np.mean([r['residual_pcc'] for r in res.values()]):+.3f}")


def summary():
    import glob
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        j = json.load(open(p)); runs.setdefault(j["model"], []).append(j)
    keys = ["pcc", "template_pcc", "residual_pcc", "pcc_in_speech", "template_in_speech"]
    print(f"{'model':10s} {'runs':>4s} " + " ".join(f"{k:>20s}" for k in keys))
    for m, js in runs.items():
        cells = []
        for k in keys:
            v = [np.nanmean([r[k] for r in j["per_subject"].values()]) for j in js]
            cells.append(f"{np.mean(v):7.3f} +- {np.std(v):5.3f}")
        print(f"{m:10s} {len(js):4d} " + " ".join(f"{c:>20s}" for c in cells))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    if args[:1] == ["summary"]:
        summary(); sys.exit()
    assert args[:1] == ["run"] and args[1] in ("baseline", "hybrid", "linear", "baseline_null", "hybrid_null"), __doc__
    NULL = args[1].endswith("_null")
    subs = ["sub-02", "sub-06"] if SMOKE else ALL_SUBS
    logf = open(os.path.join(RES, f"log_{args[1]}.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    D, ymu, ysd = load_all(subs)
    log(f"{args[1]}: {len(subs)} participants | train windows {len(windows(D, 'train'))}, val {len(windows(D, 'val'))}, test {len(windows(D, 'test'))}")
    if args[1] == "linear": run_linear(D, ymu, ysd, log)
    else: run_neural(args[1], D, ymu, ysd, log)
    summary()
