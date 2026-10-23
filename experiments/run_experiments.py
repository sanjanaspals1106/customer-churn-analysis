"""Honest model comparison: 80/20 stratified hold-out (touched once), repeated 5-fold CV on the 80%."""
import json, sys, time, warnings
sys.path.insert(0, ".")
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from src.features import load_data
from src.models import make_pipeline
warnings.filterwarnings("ignore")

X, y = load_data()
Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2024)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)

configs = [("logreg", False), ("logreg", True), ("logreg_C0.1", True), ("logreg_C0.02", True),
           ("hgb", True), ("xgb", True), ("gb", True), ("rf", True), ("soft_vote", True), ("stack", True)]
results = []
for name, eng in configs:
    t = time.time()
    m = make_pipeline(name, Xtr, engineered=eng)
    accs, aucs, f1s, lls, aps, bestacc = [], [], [], [], [], []
    for tr, te in cv.split(Xtr, ytr):
        mm = clone(m).fit(Xtr.iloc[tr], ytr.iloc[tr])
        p = mm.predict_proba(Xtr.iloc[te])[:, 1]
        yt = ytr.iloc[te]
        accs.append(accuracy_score(yt, p >= .5)); aucs.append(roc_auc_score(yt, p))
        f1s.append(f1_score(yt, p >= .5)); lls.append(log_loss(yt, p)); aps.append(average_precision_score(yt, p))
    r = dict(model=name, engineered=eng, acc=np.mean(accs), acc_sd=np.std(accs), auc=np.mean(aucs),
             f1=np.mean(f1s), logloss=np.mean(lls), pr_auc=np.mean(aps), secs=time.time() - t)
    results.append(r)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
d = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
print("majority-class baseline acc (holdout):", accuracy_score(yho, d.predict(Xho)))
json.dump(results, open("reports/cv_results.json", "w"), indent=2)
