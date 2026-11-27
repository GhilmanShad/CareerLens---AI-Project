"""Validation-only experiments; no evaluation on previously used test split."""
import argparse,gc,json,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from optimize_experience_v3 import prepare,metrics
from targeted_features import fit_extra,transform_extra

def frame(rows):return pd.DataFrame([prepare(*r) for r in rows])
def evaluate(scores,classes,y,offset=None):
    p=classes[(scores+(0 if offset is None else offset)).argmax(axis=1)]
    r=metrics(y,p);r['associate_recall']=float(np.mean(p[y=='Associate']=='Associate'));return r

def tune(scores,classes,y,ix):
    s=scores[ix]; yy=y[ix]; b=np.zeros(len(classes)); yi=np.searchsorted(classes,yy)
    def objective(o):
        pred=(s+o).argmax(axis=1);cm=np.bincount(yi*len(classes)+pred,minlength=len(classes)**2).reshape(len(classes),len(classes)); tp=cm.diagonal();den=cm.sum(axis=0)+cm.sum(axis=1);return (tp.sum()/len(yi)+np.mean(np.divide(2*tp,den,out=np.zeros(len(tp),dtype=float),where=den>0)))/2
    for step in [.15,.05,.015]:
        for i,k in enumerate(classes):
            if k=='Mid-Senior level':continue
            candidates=[]
            for delta in [-2*step,-step,0,step,2*step]:
                z=b.copy();z[i]+=delta;candidates.append((objective(z),-abs(delta),z))
            b=max(candidates,key=lambda x:x[:2])[2]
    return b

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--v3',required=True);p.add_argument('--augmented',required=True);p.add_argument('--output',default='results');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True)
    df=pd.read_csv(a.original,usecols=['job_id','title','description','company_name','formatted_experience_level'],low_memory=False).dropna(subset=['formatted_experience_level','title']).drop_duplicates('job_id').fillna(''); y=df.formatted_experience_level.str.strip().reset_index(drop=True); prefix=df.description.str.slice(0,2500);groups=pd.util.hash_pandas_object(prefix.where(prefix.ne(''),df.title.str.strip()),index=False)
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,y,groups)); tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(df.iloc[train],y.iloc[train],groups.iloc[train]));ti,vi=train[tr],train[va];assert (len(ti),len(vi),len(test))==(60295,15173,18972)
    tuning,check=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=74).split(df.iloc[vi],y.iloc[vi],groups.iloc[vi]));yt=y.iloc[ti].to_numpy();yv=y.iloc[vi].to_numpy();cols=['title','description','company_name'];trainrows=list(df.iloc[ti][cols].itertuples(index=False,name=None));valrows=list(df.iloc[vi][cols].itertuples(index=False,name=None));del df,y,prefix,groups;gc.collect()
    base=joblib.load(a.v3); aug=joblib.load(a.augmented);f=base.named_steps['features']; classes=base.classes_; print('Train/validation/tuning/check',len(yt),len(yv),len(tuning),len(check),flush=True)
    B=f.transform(frame(valrows));print('validation transformed',B.shape,flush=True)
    records=[];best=None
    def record(name,score,m,extra=None):
        nonlocal best
        for mode in ['unadjusted','adjusted']:
            offset=np.zeros(len(classes)) if mode=='unadjusted' else tune(score,classes,yv,tuning)
            r={'name':name,'mode':mode,'offsets':dict(zip(classes,offset.tolist())),'validation':evaluate(score,classes,yv,offset),'tuning':evaluate(score[tuning],classes,yv[tuning],offset),'check':evaluate(score[check],classes,yv[check],offset)};records.append(r);print('RESULT',json.dumps(r),flush=True)
            quality=(r['check']['accuracy']+r['check']['macro_f1'])/2
            if best is None or quality>best[0]:
                best=(quality,r);joblib.dump({'features':f,'classifier':m,'extra':extra,'offsets':offset},out/'selected_v4.joblib',compress=3)
            (out/'results.json').write_text(json.dumps({'results':records,'selected':best[1],'classes':classes.tolist(),'note':'Validation-only development. Offset tuning and check are disjoint description-prefix groups within previously reused validation; no test evaluation.'},indent=2))
    record('V3',base.named_steps['classifier'].decision_function(B),base.named_steps['classifier']); record('supplement_0.15',aug.named_steps['classifier'].decision_function(B),aug.named_steps['classifier']);del aug;gc.collect()
    A=f.transform(frame(trainrows));print('train transformed',A.shape,A.nnz,flush=True)
    extra,E=fit_extra(trainrows);F=transform_extra(extra,valrows);del trainrows,valrows;gc.collect();print('extra features',E.shape,E.nnz,flush=True)
    A=hstack([A,E],format='csr');B=hstack([B,F],format='csr');del E,F;gc.collect()
    for c in [.15,.3]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always',ConvergenceWarning);m=LinearSVC(C=c,dual=True,max_iter=350,tol=.003,random_state=42).fit(A,yt)
        converged=not any(issubclass(x.category,ConvergenceWarning) for x in w);print('trained',c,m.n_iter_,converged,flush=True)
        if converged:record('targeted_features_C'+str(c),m.decision_function(B),m,extra)
    print('SELECTED',best[1]['name'],best[1]['mode'],flush=True)
if __name__=='__main__':main()
