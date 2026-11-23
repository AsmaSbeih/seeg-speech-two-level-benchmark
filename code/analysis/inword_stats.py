"""Level-2 in-word analysis of inword_stageB_long.csv (per-participant models, seed 42).
Per-participant mean of word-level in-speech PCC; MR-CNN-BiLSTM vs each other decoder:
exact two-sided Wilcoxon signed-rank (full enumeration, n=10), Holm over the comparisons; bootstrap 95% CI over participants."""
import csv, sys, itertools, random
from collections import defaultdict
path = sys.argv[1]; m_min = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rows = [r for r in csv.DictReader(open(path)) if r.get('seed', '42') == '42']
vals = defaultdict(list)
for r in rows:
    if r.get('seed', '42') == '42' and int(r['n_speech_frames']) >= m_min:   # main results use seed 42
        vals[(r['system'], r['participant'])].append(float(r['inspeech_pcc']))
systems = sorted({r['system'] for r in rows}); parts = sorted({r['participant'] for r in rows})
mean = {(s, p): sum(v) / len(v) for (s, p), v in vals.items()}
nwords = sum(len(v) for (s, p), v in vals.items() if s == 'MR-CNN-BiLSTM')

def ranks(a):
    o = sorted(range(len(a)), key=lambda i: a[i]); r = [0.0] * len(a); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def wilcoxon_exact(d):
    d = [x for x in d if x != 0]; n = len(d); rk = ranks([abs(x) for x in d])
    wplus = sum(r for r, x in zip(rk, d) if x > 0); tot = sum(rk); obs = min(wplus, tot - wplus)
    cnt = 0
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(r for r, s in zip(rk, signs) if s)
        if min(w, tot - w) <= obs + 1e-12: cnt += 1
    return cnt / 2 ** n
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj
def boot_ci(d, B=10000, seed=1):
    rnd = random.Random(seed); n = len(d); ms = []
    for _ in range(B):
        s = [d[rnd.randrange(n)] for _ in range(n)]; ms.append(sum(s) / n)
    ms.sort(); return ms[int(0.025 * B)], ms[int(0.975 * B) - 1]

print(f"m_min = {m_min}; words kept (MR-CNN-BiLSTM rows) = {nwords}")
print("\nPer-participant mean word-level in-speech PCC")
print("system".ljust(18) + "".join(p[-2:].rjust(7) for p in parts) + "   mean")
for s in systems:
    v = [mean[(s, p)] for p in parts]
    print(s.ljust(18) + "".join(f"{x:7.3f}" for x in v) + f"  {sum(v)/len(v):.3f}")
ref = 'MR-CNN-BiLSTM'; others = [s for s in systems if s not in (ref, 'Timing template')]
res = []
for s in others:
    d = [mean[(ref, p)] - mean[(s, p)] for p in parts]
    res.append((s, sum(d) / len(d), sum(x > 0 for x in d), wilcoxon_exact(d), boot_ci(d)))
adj = holm([r[3] for r in res])
print(f"\nPrimary: {ref} minus each other decoder (exact two-sided Wilcoxon, n=10; Holm over {len(res)})")
for (s, D, win, p, ci), pa in zip(res, adj):
    print(f"  vs {s:12s} D={D:+.3f}  higher in {win}/10  p={p:.4f}  Holm={pa:.4f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
d = [mean[(ref, p)] - mean[('Timing template', p)] for p in parts]
print(f"\nDescriptive: {ref} minus template D={sum(d)/10:+.3f}, higher in {sum(x>0 for x in d)}/10, uncorrected p={wilcoxon_exact(d):.4f}, CI {tuple(round(x,3) for x in boot_ci(d))}")
