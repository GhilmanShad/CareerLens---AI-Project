import gc,json,warnings,argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np,pandas as pd,joblib
from scipy.sparse import hstack,vstack,save_npz
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from run_v4 import frame,evaluate
from targeted_features import transform_extra
from compare_supplement import hashes,normalize
from company_patterns import fit_patterns,pattern_scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--supplement',required=True);p.add_argument('--v4',required=True);p.add_argument('--output',default='results');p.add_argument('--patterns-only',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True)
    raw=pd.read_csv(a.original,usecols=['job_id','title','description','company_name','formatted_experience_level'],low_memory=False)
    if not a.patterns_only:
        with ZipFile(a.supplement) as z:
            with z.open('supplement_sample.csv') as f:s=pd.read_csv(f)
        s=s.dropna(subset=['job_id','job_title','description','job_level']);s=s[s.job_level.isin(['Associate','Mid-Senior level']) & s.job_title.str.strip().ne('') & s.description.str.strip().ne('')].copy();s=s[~s.job_id.isin(raw.job_id)].drop_duplicates('job_id').copy()
        for length in [500,2500]:
            ks=set(hashes(raw.description.fillna('').str.slice(0,length)));s['key']=hashes(s.description.str.slice(0,length)).values;s=s[~s.key.isin(ks)].copy()
        pairs=set(zip(raw.title.fillna('').map(normalize),raw.company_name.fillna('').map(normalize)));s=s[[p not in pairs for p in zip(s.job_title.map(normalize),s.company.fillna('').map(normalize))]].copy();s['key']=hashes(s.description.str.slice(0,500)).values;s=s[s.groupby('key').job_level.transform('nunique').eq(1)].drop_duplicates('key').copy();print('Supplement audit',s.job_level.value_counts().to_dict(),flush=True)
        s=pd.concat([d.sample(min(len(d),n),random_state=42) for label,n in [('Associate',10000),('Mid-Senior level',5000)] for d in [s[s.job_level.eq(label)]]],ignore_index=True).fillna('');ys=s.job_level.to_numpy();sr=list(s[['job_title','description','company']].itertuples(index=False,name=None));del s,pairs,ks;gc.collect()
    df=raw.dropna(subset=['formatted_experience_level','title']).drop_duplicates('job_id').fillna('');del raw;y=df.formatted_experience_level.str.strip().reset_index(drop=True);prefix=df.description.str.slice(0,2500);groups=pd.util.hash_pandas_object(prefix.where(prefix.ne(''),df.title.str.strip()),index=False)
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,y,groups));tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(df.iloc[train],y.iloc[train],groups.iloc[train]));ti,vi=train[tr],train[va];assert (len(ti),len(vi),len(test))==(60295,15173,18972)
    tuning,check=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=74).split(df.iloc[vi],y.iloc[vi],groups.iloc[vi]));yt=y.iloc[ti].to_numpy();yv=y.iloc[vi].to_numpy();cols=['title','description','company_name'];trrows=list(df.iloc[ti][cols].itertuples(index=False,name=None));vr=list(df.iloc[vi][cols].itertuples(index=False,name=None));del df,y,prefix,groups;gc.collect()
    model=joblib.load(a.v4);classes=model['classifier'].classes_;assert model['extra'] is not None
    patterns=fit_patterns(trrows,yt,classes);P=pattern_scores(patterns,vr);print('Company features: keys',len(patterns['counts']),'validation coverage',float(np.any(P!=0,axis=1).mean()),flush=True)
    def transform(rows):
        chunks=[]
        for start in range(0,len(rows),2500):
            r=rows[start:start+2500];chunks.append(hstack([model['features'].transform(frame(r)),transform_extra(model['extra'],r)],format='csr').astype(np.float32))
        return vstack(chunks,format='csr')
    B=transform(vr);print('Validation transformed',B.shape,B.nnz,flush=True)
    save_npz(out/'validation.npz',B,compressed=False);del B;gc.collect()
    np.savez(out/'labels.npz',yt=yt.astype(str),yv=yv.astype(str),ys=ys.astype(str),P=P,tuning=tuning,check=check)
    A=transform(trrows);del trrows;gc.collect();print('Original features',A.shape,A.nnz,flush=True)
    save_npz(out/'original.npz',A,compressed=False);del A;gc.collect()
    S=transform(sr);del sr;gc.collect();print('Supplement features',S.shape,S.nnz,flush=True)
    save_npz(out/'supplement.npz',S,compressed=False);del S;gc.collect()
    print('CACHE COMPLETE',flush=True)
if __name__=='__main__':main()
