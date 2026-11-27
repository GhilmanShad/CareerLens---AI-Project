"""Targeted title structure and experience features."""
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
YEAR=re.compile(r'\b(\d{1,2})(?:\s*(?:-|–|—|to)\s*(\d{1,2}))?\s*\+?\s*(?:years?|yrs?)\b',re.I)
CUES={'intern':r'\b(?:intern(?:ship)?|co[- ]?op|summer analyst)\b','entry':r'\b(?:entry[- ]level|junior|jr\.?|new grad(?:uate)?|early career|trainee|no experience)\b','associate':r'\b(?:associate|assistant|apprentice)\b','senior':r'\b(?:senior|sr\.?|principal|staff|lead|expert)\b','management':r'\b(?:manager|supervisor|head of|manage a team|direct reports|people management)\b','director':r'\b(?:director|vice president|vp|svp|avp)\b','executive':r'\b(?:chief|ceo|cfo|cto|coo|c-suite|president|executive)\b','degree':r'\b(?:bachelor|master|doctorate|phd|degree)\b','license':r'\b(?:licensed|certification|certified|license)\b','noexp':r'\b(?:no (?:prior )?experience (?:required|necessary)|will train|training provided)\b','required':r'\b(?:minimum|required|at least|must have|qualifications)\b','preferred':r'\b(?:preferred|desired|nice to have)\b'}
CUES={k:re.compile(v,re.I) for k,v in CUES.items()}
def targeted(title,desc,company):
    t=' '.join(re.findall(r'[a-z0-9]+',str(title).lower())); d=str(desc)[:14000]; c=' '.join(re.findall(r'[a-z0-9]+',str(company).lower())); w=t.split(); f={}
    if t: f['exact_title='+t]=1
    if c: f['company='+c]=1
    if t and c: f['company_title='+c+'|'+t]=1
    for n in (1,2,3):
        if len(w)>=n:
            f['first'+str(n)+'='+' '.join(w[:n])]=1
            f['last'+str(n)+'='+' '.join(w[-n:])]=1
    f['title_length='+str(min(len(w),12))]=1
    for k,p in CUES.items():
        if p.search(t): f['title_'+k]=1
        if p.search(d): f['description_'+k]=1
    for m in YEAR.finditer(d):
        before=d[max(0,m.start()-65):m.start()]; after=d[m.end():m.end()+95]; context=before+' '+after
        if not re.search(r'experience|work(?:ing)?|professional|minimum|qualifications|required|preferred',context,re.I): continue
        lo=int(m.group(1)); hi=int(m.group(2) or lo)
        if lo>40 or hi>40 or hi<lo: continue
        lo=min(lo,15); hi=min(hi,15)
        f['experience_min='+str(lo)]=1;f['experience_max='+str(hi)]=1;f['experience_bucket='+str(min(lo//2,7))]=1
        if CUES['required'].search(before):f['required_min='+str(lo)]=1
        if CUES['preferred'].search(before):f['preferred_min='+str(lo)]=1
    return f,d[3500:10000]
def build_extra(rows):
    pairs=[targeted(t,d,c) for t,d,c in rows]; return [p[0] for p in pairs],[p[1] for p in pairs]
def fit_extra(rows):
    ds,ts=build_extra(rows); dv=DictVectorizer(dtype=np.float32); tv=TfidfVectorizer(ngram_range=(1,2),min_df=5,max_features=40000,sublinear_tf=True,dtype=np.float32)
    return (dv,tv),hstack([dv.fit_transform(ds)*.5,tv.fit_transform(ts)*.8],format='csr')
def transform_extra(models,rows):
    ds,ts=build_extra(rows);dv,tv=models
    return hstack([dv.transform(ds)*.5,tv.transform(ts)*.8],format='csr')
