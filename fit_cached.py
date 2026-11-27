"""Fit a single weighted SVM in a clean process; evaluate validation only."""
import argparse,gc,json,warnings
from pathlib import Path
import joblib,numpy as np
from scipy.sparse import load_npz,vstack
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from run_v4 import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--v5',required=True);p.add_argument('--output',required=True);p.add_argument('--weight',type=float,required=True);a=p.parse_args();cache=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    data=np.load(cache/'labels.npz');yt,yv,ys,P,tuning,check=[data[k] for k in ['yt','yv','ys','P','tuning','check']]; data.close()
    A=load_npz(cache/'original.npz');S=load_npz(cache/'supplement.npz');C=vstack([A,S],format='csr');del A,S;gc.collect()
    print('TRAIN',C.shape,C.nnz,'weight',a.weight,flush=True)
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter('always',ConvergenceWarning)
        m=LinearSVC(C=.3,dual=True,max_iter=350,tol=.003,random_state=42).fit(C,np.r_[yt,ys],sample_weight=np.r_[np.ones(len(yt)),np.full(len(ys),a.weight)])
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in warn);print('FIT COMPLETE',m.n_iter_,converged,flush=True)
    del C;gc.collect();B=load_npz(cache/'validation.npz');score=m.decision_function(B);del B;gc.collect();classes=m.classes_;candidates=[]
    for w in [0,.05,.1,.2,.35]:
        s=score+w*P;r={'supplement_weight':a.weight,'pattern_weight':w,'iterations':int(m.n_iter_),'converged':converged,'validation':evaluate(s,classes,yv),'tuning':evaluate(s[tuning],classes,yv[tuning]),'check':evaluate(s[check],classes,yv[check])};candidates.append(r)
    selected=max(candidates,key=lambda r:(r['tuning']['accuracy']+r['tuning']['macro_f1'])/2);print('RESULT',json.dumps(selected),flush=True)
    tag=str(a.weight);(out/f'weight_{tag}.json').write_text(json.dumps({'selected':selected,'candidates':candidates},indent=2))
    if converged:
        model=joblib.load(a.v5);assert np.array_equal(model['classifier'].classes_,classes);model.update(classifier=m,pattern_weight=selected['pattern_weight'],offsets=np.zeros(len(classes)));joblib.dump(model,out/f'weight_{tag}.joblib',compress=3)
if __name__=='__main__':main()
