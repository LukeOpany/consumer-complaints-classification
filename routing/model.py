import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import platform
import sklearn

TEXT='Consumer complaint narrative'
LABEL='Product'


def prepare(frame, min_class=20):
    frame=frame[[TEXT,LABEL]].dropna().copy()
    frame[TEXT]=frame[TEXT].astype(str).str.replace(r'\s+',' ',regex=True).str.strip()
    frame[LABEL]=frame[LABEL].astype(str).str.strip()
    frame=frame[frame[TEXT].str.len().ge(20)&frame[LABEL].ne('')]
    # Exclude conflicting duplicate labels, then remove duplicate narratives globally.
    frame['_key']=frame[TEXT].str.casefold()
    conflicts=frame.groupby('_key')[LABEL].nunique()
    frame=frame[~frame._key.isin(conflicts[conflicts>1].index)].drop_duplicates('_key')
    counts=frame[LABEL].value_counts()
    excluded=counts[counts<min_class].to_dict()
    frame=frame[frame[LABEL].isin(counts[counts>=min_class].index)].drop(columns='_key')
    if frame[LABEL].nunique()<2: raise ValueError('Need at least two classes with enough distinct narratives')
    return frame,excluded


def choose_threshold(probabilities, labels, truth, target=0.85):
    scores=probabilities.max(axis=1); predicted=labels[probabilities.argmax(axis=1)]
    candidates=[]
    for threshold in np.arange(0.5,1.0,0.025):
        accepted=scores>=threshold
        if accepted.sum()>=20 and accuracy_score(np.asarray(truth)[accepted],predicted[accepted])>=target:
            candidates.append((int(accepted.sum()),float(threshold)))
    if not candidates: return 1.0,False
    return max(candidates,key=lambda item:(item[0],-item[1]))[1],True


def metrics(truth,predicted):
    return dict(accuracy=float(accuracy_score(truth,predicted)),macro_f1=float(f1_score(truth,predicted,average='macro',zero_division=0)),
                weighted_f1=float(f1_score(truth,predicted,average='weighted',zero_division=0)))


def train(csv_path,output='artifacts',report_dir='reports',min_class=20):
    source=Path(csv_path)
    raw=pd.read_csv(source,usecols=[TEXT,LABEL])
    frame,excluded=prepare(raw,min_class)
    train_frame,held=train_test_split(frame,test_size=0.4,stratify=frame[LABEL],random_state=42)
    validation,test=train_test_split(held,test_size=0.5,stratify=held[LABEL],random_state=42)
    pipeline=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=60000,min_df=2,sublinear_tf=True)),
                       ('classifier',SGDClassifier(loss='log_loss',class_weight='balanced',random_state=42,max_iter=1000,tol=1e-3))])
    pipeline.fit(train_frame[TEXT],train_frame[LABEL])
    baseline=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train_frame),1)),train_frame[LABEL])
    threshold,auto_enabled=choose_threshold(pipeline.predict_proba(validation[TEXT]),pipeline.classes_,validation[LABEL])
    probabilities=pipeline.predict_proba(test[TEXT]);predicted=pipeline.classes_[probabilities.argmax(axis=1)]
    accepted=(probabilities.max(axis=1)>=threshold)&auto_enabled
    report=dict(runtime=dict(python=platform.python_version(),sklearn=sklearn.__version__,pandas=pd.__version__),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_rows=len(raw),usable_rows=len(frame),
                excluded_small_classes=excluded,split_rows=dict(train=len(train_frame),validation=len(validation),test=len(test)),
                split='60/20/20 stratified random, seed 42; exact normalized duplicates removed before splitting',
                baseline=metrics(test[LABEL],baseline.predict(np.zeros((len(test),1)))),model=metrics(test[LABEL],predicted),
                per_class=classification_report(test[LABEL],predicted,output_dict=True,zero_division=0),
                labels=pipeline.classes_.tolist(),confusion_matrix=confusion_matrix(test[LABEL],predicted,labels=pipeline.classes_).tolist(),
                routing=dict(threshold=threshold,auto_enabled=auto_enabled,target_validation_accuracy=0.85,
                  test_auto_coverage=float(accepted.mean()),test_auto_accuracy=float(accuracy_score(test[LABEL].to_numpy()[accepted],predicted[accepted])) if accepted.any() else None),
                limitations=['Model scores are not calibrated probabilities.','Threshold selected on validation only; target is not a guarantee.',
                             'Random split does not measure temporal generalization; near-duplicates may remain.',
                             'Classes below minimum count are excluded. Out-of-scope products require human review.'])
    model_path=Path(output);model_path.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(pipeline=pipeline,threshold=threshold,auto_enabled=auto_enabled,source_sha256=report['source_sha256']),model_path/'routing.joblib')
    out=Path(report_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    table=pd.DataFrame(report['per_class']).T
    table.to_csv(out/'per-class.csv')
    pd.DataFrame(report['confusion_matrix'],index=pipeline.classes_,columns=pipeline.classes_).to_csv(out/'confusion-matrix.csv')
    return report


def route(bundle,text):
    if not isinstance(text,str) or len(text.strip())<20:
        return dict(product=None,score=None,decision='manual_review',reason='Provide at least 20 characters of narrative')
    probability=bundle['pipeline'].predict_proba([text])[0]
    index=int(probability.argmax());score=float(probability[index])
    accepted=bundle['auto_enabled'] and score>=bundle['threshold']
    return dict(product=str(bundle['pipeline'].classes_[index]),score=score,
                decision='suggested_route' if accepted else 'manual_review',
                reason='Above validation-selected threshold' if accepted else 'Below threshold or automatic routing disabled')
