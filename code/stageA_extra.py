"""
stageA_extra.py -- NeuroIncept and ConvED-SR(-style) in the SAME Stage-A harness as stageA_pooled.py
(same split, same windows, same loss target, same optimiser / weight decay / scheduler / early stopping / lr grid / seeds / metrics).
It imports everything from stageA_pooled.py and does not modify it.

Both models are RE-IMPLEMENTATIONS in PyTorch from the public code of the original authors, adapted to the pooled condition
(zero-padding to 127 channels + mask + subject embedding) and to our feature frames. Deviations are listed explicitly:

NeuroIncept (github owaismujtaba/NeuroInceptDecoder, Keras)
  kept   : per-frame decoder; input = 9 context frames (+-4, step 5) x channels; Inception(1x1,3,5,maxpool+1x1; 64 filters)
           -> GRU 128 -> GRU 256 -> GRU 512 (last state) -> Inception(128) -> Dense 1024-1024-512-256-128 -> linear; MSE.
  changed: (1) 80 mel bins (original 128); (2) standard tanh GRU (original Keras GRU uses relu); (3) channels zero-padded to 127 + mask;
           (4) a 32-d subject embedding is concatenated before the dense layers (pooled condition); (5) context frames beyond the
           window edge are replicated; (6) per optimisation step each window contributes 32 randomly chosen frames (all frames at
           test time), to keep CPU training feasible; the original used batch 128 and Adam 1e-4 (1e-4 is in our lr grid).

ConvED-SR-style (github yuanyu1016/ConvED-SR, PyTorch; no LICENSE file in that repository, so this is an independent re-implementation
  of the published architecture, not a copy)
  kept   : Conv1d stem (stride 2) -> 7 residual 2-D down blocks -> latent -> ConvTranspose + 4 residual up blocks with CBAM and
           4 multi-scale outputs (5x10, 10x20, 20x40, 40x80), deep-supervision loss = sum of the 4 MSEs (targets halved with
           nearest interpolation), hidden 256, latent 256; one 40-frame window (464 ms) -> 40 mel frames.
  changed: (1) input = our HG frames (one frame per mel frame) instead of 1024 Hz samples, so the encoder's last kernel is (2,1)
           instead of (2,4); (2) every 256-frame harness window is processed as 7 segments of 40 frames (starts 0,40,...,200 and 216,
           the last one end-aligned); (3) zero-padding to 127 channels + mask; (4) subject embedding added to the latent;
           (5) optimiser and schedule are the shared ones (original: Adam 5e-4, batch 256).

Usage (same terminal / notebook launcher as stageA_pooled.py):
    python stageA_extra.py smoke run neuroincept     # 1 epoch, 2 participants
    python stageA_extra.py run neuroincept
    python stageA_extra.py run convedsr
    python stageA_extra.py run neuroincept_null      # same model, noise input (1 run)
    python stageA_extra.py run convedsr_null
    python stageA_extra.py summary
"""
import os, sys, copy, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageA_pooled as A
from stageA_pooled import CH, MEL, W, BS, MAX_EP, PATIENCE, WEIGHT_DECAY, LR_FACTOR, LR_PATIENCE


# ====================================================================== NeuroIncept
class Incept(nn.Module):
    def __init__(self, cin, f):
        super().__init__()
        self.c1 = nn.Conv1d(cin, f, 1); self.c3 = nn.Conv1d(cin, f, 3, padding=1); self.c5 = nn.Conv1d(cin, f, 5, padding=2)
        self.mp = nn.MaxPool1d(3, 1, 1); self.cp = nn.Conv1d(cin, f, 1)

    def forward(self, x):                                     # (N, cin, L) -> (N, 4f, L)
        return torch.cat([F.relu(self.c1(x)), F.relu(self.c3(x)), F.relu(self.c5(x)), F.relu(self.cp(self.mp(x)))], 1)


class NeuroIncept(nn.Module):
    ORDER, STEP, K_TRAIN = 4, 5, 32

    def __init__(self, n_sub=10, emb=32):
        super().__init__()
        self.sub_embed = nn.Embedding(n_sub, emb)
        self.inc1 = Incept(CH, 64)
        self.g1 = nn.GRU(256, 128, batch_first=True); self.g2 = nn.GRU(128, 256, batch_first=True); self.g3 = nn.GRU(256, 512, batch_first=True)
        self.inc2 = Incept(512, 128)
        self.fc = nn.Sequential(nn.Linear(512 + emb, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
                                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, MEL))
        self.register_buffer("offs", torch.arange(-self.ORDER, self.ORDER + 1) * self.STEP)

    def _decode(self, seq, sid):                              # seq (N, CH, 9), sid (N,)
        h = self.inc1(seq).permute(0, 2, 1)
        h, _ = self.g1(h); h, _ = self.g2(h); h, _ = self.g3(h)
        h = self.inc2(h[:, -1].unsqueeze(-1)).flatten(1)      # last state -> (N, 512, 1) -> Inception -> (N, 512)
        return self.fc(torch.cat([h, self.sub_embed(sid)], 1))

    def _frames(self, x, t_idx):                              # x (B, CH, W); t_idx (T,) -> (B*T, CH, 9)
        idx = (t_idx[:, None] + self.offs[None, :]).clamp(0, x.size(2) - 1)       # edges replicated inside the window
        return x[:, :, idx].permute(0, 2, 1, 3).reshape(-1, x.size(1), len(self.offs))

    def forward(self, x, mask, sid):
        x = x * mask; B = x.size(0)
        out = self._decode(self._frames(x, torch.arange(W, device=x.device)), sid.repeat_interleave(W))
        return out.view(B, W, MEL).permute(0, 2, 1)

    def loss_frames(self, x, mask, sid, y):
        x = x * mask; B = x.size(0)
        t = torch.randperm(W, device=x.device)[:self.K_TRAIN]
        out = self._decode(self._frames(x, t), sid.repeat_interleave(len(t))).view(B, len(t), MEL).permute(0, 2, 1)
        return F.mse_loss(out, y[:, :, t])


# ====================================================================== ConvED-SR-style
class CBAM(nn.Module):
    def __init__(self, c, reduction=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // reduction, 1), nn.ReLU(inplace=True), nn.Conv2d(c // reduction, c, 1))
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 3, 1, 1, bias=False), nn.BatchNorm2d(1, momentum=0.01), nn.Sigmoid())

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        return x * self.sa(torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1))


class ResDown(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout // 2, 3, 2, 1); self.bn1 = nn.BatchNorm2d(cout // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(cout // 2, cout, 3, 1, 1); self.bn2 = nn.BatchNorm2d(cout, eps=1e-4)
        self.skip = nn.Conv2d(cin, cout, 1, 2, 0)

    def forward(self, x):
        s = self.skip(x)
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))) + s))


class ResUp(nn.Module):
    def __init__(self, cin, cout, pad_row=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cin // 2, 3, 1, 1); self.bn1 = nn.BatchNorm2d(cin // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(cin // 2, cout, 3, 1, 1); self.bn2 = nn.BatchNorm2d(cout, eps=1e-4)
        self.skip = nn.Conv2d(cin, cout, 3, 1, 1)
        self.pad_row = pad_row
        self.cbam = CBAM(cout); self.out = nn.Conv2d(cout, 1, 3, 1, 1)

    def forward(self, x, prev=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.pad_row:
            x = F.pad(x, (0, 0, 0, 1))                        # one extra row at the bottom (time axis): 4 -> 5
        s = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + s)
        o = self.out(self.cbam(x))
        if prev is not None:
            o = (o + prev) / 2
        return x, o


class ConvEDSR(nn.Module):
    SEG = 40
    STARTS = list(range(0, W - 40 + 1, 40)) + [W - 40]        # 0,40,...,200 and the end-aligned 216

    def __init__(self, n_sub=10, emb=32, hidden=256, latent=256):
        super().__init__()
        self.sub_embed = nn.Embedding(n_sub, emb); self.sub_to_latent = nn.Linear(emb, latent)
        self.conv_in = nn.Conv1d(CH, hidden, 3, 2, 1)
        ch = [1] + [latent // d for d in (128, 64, 32, 16, 8, 4, 2)]
        self.down = nn.ModuleList([ResDown(ch[i], ch[i + 1]) for i in range(7)])
        wl = self._final_width()
        self.conv_out = nn.Conv2d(latent // 2, latent, (hidden // 128, wl))
        self.conv_t = nn.ConvTranspose2d(latent, latent // 2, (2, 5), 1)
        self.up1 = ResUp(latent // 2, latent // 4, pad_row=True); self.up2 = ResUp(latent // 4, latent // 8)
        self.up3 = ResUp(latent // 8, latent // 16); self.up4 = ResUp(latent // 16, latent // 32)
        self.hidden = hidden

    def _final_width(self):
        w = (self.SEG + 2 - 3) // 2 + 1
        for _ in range(7):
            w = (w + 2 - 3) // 2 + 1
        return w

    def _segments(self, x):                                   # x (B, C, W) -> (B*S, C, 40)
        return torch.stack([x[:, :, s:s + self.SEG] for s in self.STARTS], 1).flatten(0, 1)

    def _net(self, seg, sid):                                 # seg (N, CH, 40), sid (N,)
        z = F.relu(self.conv_in(seg)).unsqueeze(1)            # (N, 1, hidden, 20)
        for blk in self.down:
            z = blk(z)
        z = self.conv_out(z) + self.sub_to_latent(self.sub_embed(sid))[:, :, None, None]
        z = F.relu(self.conv_t(z))
        z, o1 = self.up1(z); z, o2 = self.up2(z, F.interpolate(o1, scale_factor=2)); z, o3 = self.up3(z, F.interpolate(o2, scale_factor=2))
        z, o4 = self.up4(z, F.interpolate(o3, scale_factor=2))
        return [o.squeeze(1) for o in (o1, o2, o3, o4)]      # (N,5,10) (N,10,20) (N,20,40) (N,40,80)

    def forward(self, x, mask, sid):
        B = x.size(0); S = len(self.STARTS)
        o4 = self._net(self._segments(x * mask), sid.repeat_interleave(S))[3].view(B, S, self.SEG, MEL)
        P = x.new_zeros(B, W, MEL)
        for i, s in enumerate(self.STARTS[:-1]):
            P[:, s:s + self.SEG] = o4[:, i]
        last = self.STARTS[-1]; first_uncovered = self.STARTS[-2] + self.SEG
        P[:, first_uncovered:] = o4[:, -1, first_uncovered - last:]
        return P.permute(0, 2, 1)

    def loss_frames(self, x, mask, sid, y):                   # deep supervision on the 4 scales, targets halved by nearest interpolation
        B = x.size(0); S = len(self.STARTS)
        outs = self._net(self._segments(x * mask), sid.repeat_interleave(S))
        ys = torch.stack([y.permute(0, 2, 1)[:, s:s + self.SEG] for s in self.STARTS], 1).flatten(0, 1)      # (N, 40, 80)
        y2 = F.interpolate(ys.unsqueeze(1), scale_factor=0.5).squeeze(1)
        y1 = F.interpolate(y2.unsqueeze(1), scale_factor=0.5).squeeze(1)
        y0 = F.interpolate(y1.unsqueeze(1), scale_factor=0.5).squeeze(1)
        return sum(F.mse_loss(o, t) for o, t in zip(outs, (y0, y1, y2, ys)))


# ====================================================================== shared training function (identical to stageA_pooled.train_model
# except that a model may provide loss_frames())
def train_model(name, lr, seed, D, epochs, log):
    torch.manual_seed(seed); np.random.seed(seed)
    model = A.CTORS[name.replace('_null', '')]().to(A.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', LR_FACTOR, patience=LR_PATIENCE)
    tr, va = A.windows(D, "train"), A.windows(D, "val")
    best, best_sd, bad = np.inf, None, 0
    for ep in range(epochs):
        t0 = time.time(); model.train(); perm = np.random.permutation(len(tr)); tot = n = 0
        for i in range(0, len(perm), BS):
            ii = perm[i:i + BS]
            if len(ii) < 2: continue
            x, m, y, sid = A.make_batch(D, tr, ii)
            loss = model.loss_frames(x, m, sid, y) if hasattr(model, "loss_frames") else F.mse_loss(model(x, m, sid), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(ii); n += len(ii)
        v = A.val_mse(model, D, va); sched.step(v)
        log(f"    [{name} lr={lr:g} seed={seed}] epoch {ep + 1}/{epochs}: train loss {tot / n:.4f} | val MSE {v:.4f} | {time.time() - t0:.0f}s")
        if v < best - 1e-5:
            best, best_sd, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_sd)
    return model, best


A.CTORS.update({"neuroincept": NeuroIncept, "convedsr": ConvEDSR})
A.train_model = train_model                                    # run_neural() looks the function up at call time

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    if args[:1] == ["summary"]:
        A.summary(); sys.exit()
    names = ("neuroincept", "convedsr", "neuroincept_null", "convedsr_null")
    assert args[:1] == ["run"] and args[1] in names, __doc__
    A.NULL = args[1].endswith("_null")
    subs = ["sub-02", "sub-06"] if A.SMOKE else A.ALL_SUBS
    logf = open(os.path.join(A.RES, f"log_{args[1]}.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    D, ymu, ysd = A.load_all(subs)
    log(f"{args[1]}: {len(subs)} participants | params {sum(p.numel() for p in A.CTORS[args[1].replace('_null', '')]().parameters()):,}")
    A.run_neural(args[1], D, ymu, ysd, log)
    A.summary()
