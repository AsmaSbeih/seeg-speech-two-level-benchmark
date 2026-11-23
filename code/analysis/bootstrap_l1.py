"""Level-1 bootstrap over participants (seed 42 runs). Per-participant means over the 10 folds, then 10,000 resamples of the
10 participants with replacement (numpy default_rng(2026), rng.integers(0, 10, size=(10000, 10)), the same draws for every
statistic); 95% interval = 2.5th and 97.5th percentiles (numpy default, linear interpolation).
Usage: python3 -I bootstrap_l1.py stageB_long.csv"""
import csv, sys
from collections import defaultdict
import numpy as np
v=defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    s,m=r['subject'],r['model']
    for met,col in (('pcc','pcc'),('ins','pcc_in_speech'),('res','residual_pcc')): v[(met,m,s)].append(float(r[col]))
    if m=='baseline':
        v[('pcc','template',s)].append(float(r['template_pcc'])); v[('ins','template',s)].append(float(r['template_in_speech']))
subs=sorted({k[2] for k in v})
P=lambda met,m: np.array([np.mean(v[(met,m,s)]) for s in subs])
idx=np.random.default_rng(2026).integers(0,10,size=(10000,10))
def ci(x): b=x[idx].mean(1); return x.mean(), *np.percentile(b,[2.5,97.5])
names=[('hybrid','MR-CNN-BiLSTM'),('baseline','CNN-BiLSTM'),('linear','Linear'),('neuroincept','NeuroIncept'),('convedsr','ConvED-SR'),('template','Template')]
print('MEANS')
for m,n in names:
    print(n, *[('%.3f [%.3f, %.3f]'%ci(P(met,m)) if not (m=='template' and met=='res') else '--') for met in ('pcc','ins','res')], sep=' | ')
print('DIFFS')
pairs=[('hybrid',o) for o in ('baseline','linear','neuroincept','convedsr','template')]+[(m,'template') for m in ('linear','baseline','neuroincept','convedsr')]
nm=dict(names)
for a,b in pairs:
    out=[]
    for met in ('pcc','ins','res'):
        if b=='template' and met=='res': out.append('--'); continue
        out.append('%+.3f [%+.3f, %+.3f]'%ci(P(met,a)-P(met,b)))
    print(f'{nm[a]} - {nm[b]}', *out, sep=' | ')
