"""Level 2 per seed (42, 123, 456) and on participant values averaged over the 3 seeds.
Words with >= 20 speech frames; participant mean over words; exact two-sided Wilcoxon (n=10); Holm over the 4 comparisons
of the decoder with the highest mean against the other four. Linear and template are deterministic (seed 42 used for every seed).
Usage: python3 -I inword_seeds.py inword_stageB_long_2.csv"""
import csv, sys, itertools
from collections import defaultdict
def wil(d):
    d=[x for x in d if x!=0]; n=len(d); a=[abs(x) for x in d]; o=sorted(range(n),key=lambda i:a[i]); rk=[0]*n; i=0
    while i<n:
        j=i
        while j+1<n and a[o[j+1]]==a[o[i]]: j+=1
        for k in range(i,j+1): rk[o[k]]=(i+j)/2+1
        i=j+1
    tot=sum(rk); wp=sum(r for r,x in zip(rk,d) if x>0); obs=min(wp,tot-wp)
    return sum(min(w,tot-w)<=obs+1e-12 for w in (sum(r for r,b in zip(rk,s) if b) for s in itertools.product([0,1],repeat=n)))/2**n
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); out=[0]*len(ps); m=0
    for r,i in enumerate(o): m=max(m,min(1,(len(ps)-r)*ps[i])); out[i]=m
    return out
M=lambda x:sum(x)/len(x)
v=defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if int(r['n_speech_frames'])>=20: v[(r['seed'],r['system'],r['participant'])].append(float(r['inspeech_pcc']))
parts=sorted({k[2] for k in v}); dec=['MR-CNN-BiLSTM','CNN-BiLSTM','Linear','NeuroIncept','ConvED-SR']
def pv(seed,s):
    sd='42' if s in ('Linear','Timing template') else seed
    return {p:M(v[(sd,s,p)]) for p in parts}
variants={'42':lambda s:pv('42',s),'123':lambda s:pv('123',s),'456':lambda s:pv('456',s),
          'mean3':lambda s:{p:M([pv(x,s)[p] for x in ('42','123','456')]) for p in parts}}
for name,f in variants.items():
    P={s:f(s) for s in dec+['Timing template']}
    means={s:M(list(P[s].values())) for s in P}
    best=max(dec,key=lambda s:means[s])
    res=[]
    for o in dec:
        if o==best: continue
        d=[P[best][p]-P[o][p] for p in parts]; res.append((o,M(d),sum(x>0 for x in d),wil(d)))
    h=holm([x[-1] for x in res])
    print(f"== {name}: means", {s:round(m,3) for s,m in means.items()}, "| highest:",best)
    for r,hh in zip(res,h): print(f"   {best} - {r[0]:11s} d {r[1]:+.3f} n {r[2]}/10 p {r[3]:.4f} pH {hh:.4f}")
