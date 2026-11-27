"""CareerLens experience-level classification: grouped validation tuning."""
import gc, json, re, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

EXPERIENCE = re.compile(r'\b(\d{1,2})\s*(?:\+|(?:-|–|to)\s*\d{1,2})?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:[\w-]+\s+){0,3}experience\b', re.I)

def prepare(title, description, company=''):
    description = str(description or '')[:3500]
    title = str(title or '').strip()
    yrs = [int(x) for x in EXPERIENCE.findall(description) if int(x) <= 40]
    tokens = ['years_' + str(min(n, 15)) for n in yrs]
    if yrs:
        tokens += ['minyears_' + str(min(min(yrs),15)), 'maxyears_' + str(min(max(yrs),15))]
    c = re.sub(r'\W+', '_', str(company or '').lower()).strip('_')
    if c:
        tokens += ['company_' + c]
        tokens += ['co_' + c + '_' + w for w in re.findall(r'\b\w+\b',title.lower())]
    return {'title':title,'description':description,'context':' '.join(tokens)}

def metrics(y,p):
    pr,rc,f,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    wf=precision_recall_fscore_support(y,p,average='weighted',zero_division=0)[2]
    return dict(accuracy=accuracy_score(y,p),macro_precision=pr,macro_recall=rc,macro_f1=f,weighted_f1=wf)

def main():
    start=time.monotonic()
    df=pd.read_csv('archive/postings.csv',usecols=['job_id','title','description','company_name','formatted_experience_level'],low_memory=False).dropna(subset=['formatted_experience_level','title']).drop_duplicates('job_id').fillna('')
    y=df.formatted_experience_level.str.strip().reset_index(drop=True)
    prefix=df.description.str.slice(0,2500)
    groups=pd.util.hash_pandas_object(prefix.where(prefix.ne(''),df.title.str.strip()),index=False)
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,y,groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(df.iloc[train],y.iloc[train],groups.iloc[train])); a,b=train[tr],train[va]
    X=pd.DataFrame([prepare(t,d,c) for t,d,c in zip(df.title,df.description,df.company_name)])
    del df,prefix,groups; gc.collect()
    print('Train, validation, previously evaluated test:',len(a),len(b),len(test),flush=True)
    features=ColumnTransformer([
        ('title',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,dtype=np.float32,sublinear_tf=True),'title'),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=60000,dtype=np.float32,sublinear_tf=True),'title'),
        ('description',TfidfVectorizer(ngram_range=(1,2),min_df=5,max_features=50000,sublinear_tf=True,dtype=np.float32),'description'),
        ('context',TfidfVectorizer(min_df=2,dtype=np.float32),'context')],transformer_weights={'title':1.5,'char':1,'description':1.3,'context':.8})
    A=features.fit_transform(X.iloc[a]); B=features.transform(X.iloc[b]); print('Features',A.shape,'nnz',A.nnz,'seconds',time.monotonic()-start,flush=True)
    best=None; records=[]
    for c,w in [(0.3,None),(1.,None),(1.,{'Associate':1.6,'Director':1.3,'Executive':1.5,'Internship':1.2,'Entry level':1.,'Mid-Senior level':1.}),(1.,'balanced')]:
        t=time.monotonic()
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always',ConvergenceWarning)
            m=LinearSVC(C=c,class_weight=w,dual=True,max_iter=350,tol=.003,random_state=42).fit(A,y.iloc[a])
        p=m.predict(B); r=metrics(y.iloc[b],p); r.update(C=c,class_weight=w,iterations=int(m.n_iter_),converged=not any(issubclass(x.category,ConvergenceWarning) for x in warn),seconds=time.monotonic()-t)
        print('Validation',json.dumps(r),flush=True); records.append(r)
        score=(r['accuracy']+r['macro_f1'])/2
        if best is None or score>best[0]:
            best=(score,m,r)
            joblib.dump(Pipeline([('features',features),('classifier',m)]),'models/experience_svm_v3.joblib',compress=3)
        Path('reports/optimization_v3_validation.json').write_text(json.dumps(records,indent=2))
    del A,B; gc.collect()
    print('Selected',best[2],flush=True)
    pipe=Pipeline([('features',features),('classifier',best[1])]); p=pipe.predict(X.iloc[test]); r=metrics(y.iloc[test],p)
    report=classification_report(y.iloc[test],p,digits=4,zero_division=0)
    print('HELD-OUT RETEST',json.dumps(r),flush=True); print(report,flush=True); print(confusion_matrix(y.iloc[test],p),flush=True)
    Path('reports/optimization_v3_result.json').write_text(json.dumps({'selected':best[2],'retest':r,'classification_report':report,'confusion_matrix':confusion_matrix(y.iloc[test],p).tolist(),'classes':best[1].classes_.tolist(),'note':'Previously evaluated test split; not an untouched final holdout.'},indent=2))
if __name__=='__main__': main()
