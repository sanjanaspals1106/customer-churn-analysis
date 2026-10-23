"""Train the final churn model and write model.pkl + reports/metrics.json.

Protocol: 20% stratified hold-out is evaluated exactly once. The decision threshold is picked on
out-of-fold predictions of the 80% development split (never on the hold-out). The shipped model is
then refit on all rows.   Run: python train.py
"""
import json, warnings
import joblib, numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, brier_score_loss)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from src.features import load_data
from src.models import make_pipeline

warnings.filterwarnings("ignore")
MODEL = "soft_vote"

X, y = load_data()
Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2024)
pipe = make_pipeline(MODEL, Xtr)

oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                        method="predict_proba")[:, 1]
grid = np.arange(0.2, 0.7, 0.01)
thr = float(grid[np.argmax([f1_score(ytr, oof >= t) for t in grid])])

fitted = clone(pipe).fit(Xtr, ytr)
p = fitted.predict_proba(Xho)[:, 1]

def summarize(t):
    pred = p >= t
    tn, fp, fn, tp = confusion_matrix(yho, pred).ravel()
    return dict(threshold=float(t), accuracy=accuracy_score(yho, pred), precision=precision_score(yho, pred),
                recall=recall_score(yho, pred), f1=f1_score(yho, pred), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

# bootstrap CI on hold-out accuracy at 0.5
rng = np.random.default_rng(0); yv = yho.values
boots = [accuracy_score(yv[i], p[i] >= .5) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(1000))]
metrics = dict(model=MODEL, n_train=len(Xtr), n_holdout=len(Xho),
               majority_baseline_accuracy=float(1 - yho.mean()),
               roc_auc=roc_auc_score(yho, p), pr_auc=average_precision_score(yho, p), brier=brier_score_loss(yho, p),
               at_0_5=summarize(0.5), at_tuned=summarize(thr),
               accuracy_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
imp = permutation_importance(fitted, Xho, yho, scoring="roc_auc", n_repeats=10, random_state=0)
metrics["permutation_importance"] = dict(sorted(zip(X.columns, imp.importances_mean.round(5).tolist()),
                                                key=lambda kv: -kv[1]))
json.dump(metrics, open("reports/metrics.json", "w"), indent=2, default=float)

final = clone(pipe).fit(X, y)
joblib.dump(dict(pipeline=final, threshold=thr, metrics=metrics, columns=list(X.columns)), "model.pkl")
print(json.dumps({k: v for k, v in metrics.items() if k != "permutation_importance"}, indent=1, default=float))
print("top features:", list(metrics["permutation_importance"].items())[:6])
