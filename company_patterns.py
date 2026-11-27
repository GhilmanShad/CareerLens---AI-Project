import re
from collections import defaultdict
import numpy as np

def norm(s):return ' '.join(re.findall(r'\w+',str(s).casefold()))
def keys(row):
    t,d,c=row;t=norm(t);c=norm(c);w=t.split();out=[]
    if c and c not in ('unknown','nan'):
        out.append(('company',c))
        if t:out.append(('company_title',c,t))
        for n in (1,2):
            if len(w)>=n:
                out.append(('company_suffix',c,' '.join(w[-n:])))
                out.append(('company_prefix',c,' '.join(w[:n])))
    return list(set(out))
def fit_patterns(rows,y,classes):
    cnt=defaultdict(lambda:np.zeros(len(classes),dtype=np.int32));yi=np.searchsorted(classes,y)
    for row,i in zip(rows,yi):
        for k in keys(row):cnt[k][i]+=1
    prior=np.bincount(yi,minlength=len(classes)).astype(float);prior/=prior.sum()
    return {'counts':{k:v for k,v in cnt.items() if v.sum()>=3},'prior':prior}
def pattern_scores(model,rows):
    prior=model['prior'];out=np.zeros((len(rows),len(prior)))
    for i,row in enumerate(rows):
        vals=[];weights=[]
        for k in keys(row):
            v=model['counts'].get(k)
            if v is None:continue
            n=v.sum();p=(v+8*prior)/(n+8)
            vals.append(np.clip(np.log(p/prior),-2,2));weights.append((2 if k[0]=='company_title' else .6 if k[0]=='company' else 1)*n/(n+8))
        if vals:out[i]=np.average(vals,axis=0,weights=weights)
    return out
