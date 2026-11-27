"""CareerLens V3 vs EMSCAD augmentation; validation-only selection."""
import argparse
import gc
import json
import re
import unicodedata
import warnings
from pathlib import Path
from zipfile import ZipFile
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from optimize_experience_v3 import prepare, metrics

LABELS = {s.casefold(): s for s in ['Associate', 'Director', 'Entry level', 'Executive', 'Internship', 'Mid-Senior level']}

def normalize(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s)).casefold()).strip()

def hashes(s):
    return pd.util.hash_pandas_object(s.map(normalize), index=False)

def read_supplement(path):
    columns = ['title','description','requirements','required_experience','fraudulent']
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as z:
            names = [n for n in z.namelist() if Path(n).name == 'fake_job_postings.csv']
            if len(names) != 1:
                raise ValueError('Expected exactly one fake_job_postings.csv in ZIP')
            with z.open(names[0]) as f:
                return pd.read_csv(f, usecols=columns)
    return pd.read_csv(path, usecols=columns)

def clean_supplement(raw, original):
    d = raw.copy()
    stats = {'raw':len(d)}
    d = d[pd.to_numeric(d.fraudulent,errors='coerce').eq(0)].copy()
    stats['genuine']=len(d)
    d['label']=d.required_experience.fillna('').map(normalize).map(LABELS)
    d=d.dropna(subset=['label','title','description']).copy()
    d=d[d.title.str.strip().ne('') & d.description.str.strip().ne('')].copy()
    stats['valid_labeled']=len(d)
    # Remove normalized matching description prefixes across ALL original partitions.
    # This catches exact/prefix duplicates, not all paraphrased or near duplicates.
    original_keys=set(hashes(original.description.fillna('').str.slice(0,2500)))
    d['key']=hashes(d.description.str.slice(0,2500)).values
    d=d[~d.key.isin(original_keys)].copy()
    stats['after_original_overlap_filter']=len(d)
    # Drop conflicting label groups rather than select an arbitrary label.
    counts=d.groupby('key').label.transform('nunique')
    d=d[counts.eq(1)].drop_duplicates('key').copy()
    stats['usable']=len(d)
    stats['class_distribution']=d.label.value_counts().to_dict()
    # Requirements are legitimate text, but no reliable company-name field exists.
    d['description']=d.description.str.cat(d.requirements.fillna(''),sep='\n')
    d['company_name']=''
    return d, stats

def feature_frame(d):
    return pd.DataFrame([prepare(t,desc,c) for t,desc,c in zip(d.title,d.description,d.company_name)])

def build_features():
    return ColumnTransformer([
        ('title',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,dtype=np.float32,sublinear_tf=True),'title'),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=60000,dtype=np.float32,sublinear_tf=True),'title'),
        ('description',TfidfVectorizer(ngram_range=(1,2),min_df=5,max_features=50000,sublinear_tf=True,dtype=np.float32),'description'),
        ('context',TfidfVectorizer(min_df=2,dtype=np.float32),'context')],transformer_weights={'title':1.5,'char':1,'description':1.3,'context':.8})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('supplement',type=Path)
    p.add_argument('--original',type=Path,default=Path('archive/postings.csv'))
    p.add_argument('--output',type=Path,default=Path('supplement_results'))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.supplement.is_file(): p.error('Supplement CSV or ZIP not found')
    raw=pd.read_csv(args.original,usecols=['job_id','title','description','company_name','formatted_experience_level'],low_memory=False)
    supp,audit=clean_supplement(read_supplement(args.supplement),raw)
    print('Supplement audit:',json.dumps(audit),flush=True)
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    if supp.empty: raise ValueError('No compatible supplementary records')
    df=raw.dropna(subset=['formatted_experience_level','title']).drop_duplicates('job_id').fillna('')
    del raw
    y=df.formatted_experience_level.str.strip().reset_index(drop=True)
    prefix=df.description.str.slice(0,2500)
    groups=pd.util.hash_pandas_object(prefix.where(prefix.ne(''),df.title.str.strip()),index=False)
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,y,groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(df.iloc[train],y.iloc[train],groups.iloc[train])); a,b=train[tr],train[va]
    assert not set(groups.iloc[a]) & set(groups.iloc[b])
    assert not set(groups.iloc[train]) & set(groups.iloc[test])
    X=feature_frame(df.iloc[a]); V=feature_frame(df.iloc[b]); S=feature_frame(supp)
    ya=y.iloc[a].reset_index(drop=True); yv=y.iloc[b].reset_index(drop=True)
    ys=supp.label.reset_index(drop=True)
    del df,prefix,groups,supp; gc.collect()
    results=[]
    best=None
    print('Original train/validation/reserved test:',len(a),len(b),len(test),flush=True)
    for name,xx,yy in [('original',X,ya),('augmented',pd.concat([X,S],ignore_index=True),pd.concat([ya,ys],ignore_index=True))]:
        f=build_features(); A=f.fit_transform(xx); B=f.transform(V)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always',ConvergenceWarning)
            m=LinearSVC(C=.3,dual=True,max_iter=350,tol=.003,random_state=42).fit(A,yy)
        pred=m.predict(B); r=metrics(yv,pred); r.update(configuration=name,training_records=len(yy),validation_records=len(yv),converged=not any(issubclass(i.category,ConvergenceWarning) for i in w),iterations=int(m.n_iter_),classification_report=classification_report(yv,pred,output_dict=True,zero_division=0))
        print(name,json.dumps(r),flush=True); results.append(r)
        score=(r['accuracy']+r['macro_f1'])/2
        if r['converged'] and (best is None or score>best[0]):
            best=(score,name)
            joblib.dump(Pipeline([('features',f),('classifier',m)]),args.output/'selected_experimental_svm.joblib',compress=3)
        del A,B,f,m; gc.collect()
        (args.output/'validation_results.json').write_text(json.dumps({'results':results,'selected':best[1] if best else None,'selection':'mean of accuracy and macro F1, converged candidates only','note':'Validation comparison only; previously evaluated test partition not used. Fresh external test needed.'},indent=2))

if __name__=='__main__': main()
