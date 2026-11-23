# Level-1 and level-2 results with paper folds excluded (learning-rate selection used blocks 1 and 6). Usage: python3 -I lr_fold_exclusion.py stageB_long.csv inword_stageB_long.csv "1,6"
import csv, sys, itertools
from collections import defaultdict
L1, L2, excl = sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3].split(',')} if sys.argv[3] else set()
def wil(d):
    d=[x for x in d if x!=0]; n=len(d); a=[abs(x) for x in d]; o=sorted(range(n),key=lambda i:a[i]); rk=[0]*n; i=0
    while i<n:
        j=i
        while j+1<n and a[o[j+1]]==a[o[i]]: j+=1
        for k in range(i,j+1): rk[o[k]]=(i+j)/2+1
        i=j+1
    tot=sum(rk); wp=sum(r for r,x in zip(rk,d) if x>0); obs=min(wp,tot-wp); c=0
    for s in itertools.product([0,1],repeat=n):
        w=sum(r for r,b in zip(rk,s) if b); c+= min(w,tot-w)<=obs+1e-12
    return c/2**n
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); out=[0]*len(ps); m=0
    for r,i in enumerate(o): m=max(m,min(1,(len(ps)-r)*ps[i])); out[i]=m
    return out
# level 1 (file folds 0-9 -> paper folds 1-10)
v=defaultdict(list); t=defaultdict(list); vs=defaultdict(list); ts=defaultdict(list)
for r in csv.DictReader(open(L1)):
    if int(r['fold'])+1 in excl: continue
    k=(r['model'],r['subject']); v[k].append(float(r['pcc'])); t[k].append(float(r['template_pcc']))
    vs[k].append(float(r['pcc_in_speech'])); ts[k].append(float(r['template_in_speech']))
M=lambda x:sum(x)/len(x)
mods=['linear','neuroincept','convedsr','baseline','hybrid']; subs=sorted({s for _,s in v})
res=[]; 
for met,(A,B) in [('PCC',(v,t)),('inPCC',(vs,ts))]:
    for m in mods:
        d=[M(A[(m,s)])-M(B[(m,s)]) for s in subs]; res.append((met,m,M([M(A[(m,s)]) for s in subs]),M([M(B[(m,s)]) for s in subs]),M(d),sum(x>0 for x in d),wil(d)))
ph=holm([r[-1] for r in res])
print('LEVEL 1, excluded paper folds',sorted(excl),'folds/participant',10-len(excl))
for r,h in zip(res,ph): print(f'  {r[0]:5s} {r[1]:11s} sys {r[2]:.3f} tem {r[3]:.3f} d {r[4]:+.3f} n {r[5]}/10 p {r[6]:.4f} pH {h:.3f}')
# level 2: check fold numbering in inword file
w=defaultdict(list)
for r in csv.DictReader(open(L2)):
    if r.get('seed','42')!='42' or int(r['n_speech_frames'])<20: continue   # seed 42 only
    f=int(r['fold']); pf=f+1
    if pf in excl: continue
    w[(r['system'],r['participant'])].append(float(r['inspeech_pcc']))
parts=sorted({p for _,p in w}); mm={k:M(x) for k,x in w.items()}
nw=sum(len(x) for (s,p),x in w.items() if s=='MR-CNN-BiLSTM')
print('LEVEL 2 words',nw, {s:round(M([mm[(s,p)] for p in parts]),3) for s in ['MR-CNN-BiLSTM','CNN-BiLSTM','Linear','NeuroIncept','ConvED-SR','Timing template']})
oth=['CNN-BiLSTM','Linear','NeuroIncept','ConvED-SR']; out=[]
for o in oth:
    d=[mm[('MR-CNN-BiLSTM',p)]-mm[(o,p)] for p in parts]; out.append((o,M(d),sum(x>0 for x in d),wil(d)))
for r,h in zip(out,holm([x[-1] for x in out])): print(f'  vs {r[0]:11s} d {r[1]:+.3f} n {r[2]}/10 p {r[3]:.4f} pH {h:.4f}')
d=[mm[('MR-CNN-BiLSTM',p)]-mm[('Timing template',p)] for p in parts]; print(f'  vs template d {M(d):+.3f} n {sum(x>0 for x in d)}/10 (descriptive)')
