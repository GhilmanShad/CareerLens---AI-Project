"""Inference for the frozen CareerLens V6 experience classifier."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from optimize_experience_v3 import prepare
from targeted_features import transform_extra
from company_patterns import pattern_scores

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'experience_svm_v6.joblib'

def load_model(path=MODEL_PATH):
    # Only load the trusted, locally supplied project artifact.
    return joblib.load(path)

def predict_level(model, title, description='', company=''):
    title = str(title or '').strip()
    if not title:
        raise ValueError('Enter a job title.')
    rows = [(title, str(description or ''), str(company or ''))]
    X = hstack([model['features'].transform(pd.DataFrame([prepare(*rows[0])])),
                transform_extra(model['extra'], rows)], format='csr').astype(np.float32)
    classifier = model['classifier']
    scores = classifier.decision_function(X) + model['pattern_weight'] * pattern_scores(model['patterns'], rows) + model['offsets']
    return str(classifier.classes_[scores.argmax(axis=1)[0]])
