"""Retest frozen V6 on the original, previously evaluated group holdout."""
import argparse,json,gc,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix
from scipy.sparse import hstack
from run_v4 import frame,evaluate
from targeted_features import transform_extra
from company_patterns import pattern_scores

def main():
 p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 df=pd.read_csv(a.original,usecols=['job_id','title','description','company_name','formatted_experience_level'],low_memory=False).dropna(subset=['formatted_experience_level','title']).drop_duplicates('job_id').fillna('');y=df.formatted_experience_level.str.strip().reset_index(drop=True);prefix=df.description.str.slice(0,2500);groups=pd.util.hash_pandas_object(prefix.where(prefix.ne(''),df.title.str.strip()),index=False)
 train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,y,groups));tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(df.iloc[train],y.iloc[train],groups.iloc[train]));assert (len(tr),len(va),len(test))==(60295,15173,18972);assert not set(groups.iloc[train])&set(groups.iloc[test]);yt=y.iloc[test].to_numpy();rows=list(df.iloc[test][['title','description','company_name']].itertuples(index=False,name=None));del df,y,groups,prefix;gc.collect()
 model=joblib.load(a.model);m=model['classifier'];scores=[]
 for i in range(0,len(rows),1000):
  r=rows[i:i+1000];X=hstack([model['features'].transform(frame(r)),transform_extra(model['extra'],r)],format='csr').astype(np.float32);scores.append(m.decision_function(X)+model['pattern_weight']*pattern_scores(model['patterns'],r)+model['offsets']);print('Scored',min(i+1000,len(rows)),flush=True)
 s=np.vstack(scores);pred=m.classes_[s.argmax(axis=1)];result={'note':'Frozen selected V6 (supplement weight 0.10) retest on original previously evaluated holdout; not a fresh external holdout. No tuning performed.','model_sha256':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),'test_records':len(yt),'training_original':len(tr),'training_supplement':15000,'validation_records':len(va),'metrics':evaluate(s,m.classes_,yt),'classes':m.classes_.tolist(),'confusion_matrix':confusion_matrix(yt,pred,labels=m.classes_).tolist(),'classification_report':classification_report(yt,pred,labels=m.classes_,output_dict=True,zero_division=0)}
 Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
