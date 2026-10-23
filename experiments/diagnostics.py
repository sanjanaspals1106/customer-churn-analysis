"""Evidence for the accuracy ceiling: learning curve + train-vs-CV gap on the 80% development split."""
import json, sys, warnings
sys.path.insert(0, ".")
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.features import load_data
from src.models import make_pipeline
warnings.filterwarnings("ignore")

X, y = load_data()
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2024)
out = {}
for name in ["logreg_C0.1", "xgb"]:
    rows = []
    for frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        a, t = [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=3).split(Xtr, ytr):
            n = int(len(tr) * frac); idx = tr[:n]
            m = clone(make_pipeline(name, Xtr)).fit(Xtr.iloc[idx], ytr.iloc[idx])
            a.append(accuracy_score(ytr.iloc[te], m.predict(Xtr.iloc[te])))
            t.append(accuracy_score(ytr.iloc[idx], m.predict(Xtr.iloc[idx])))
        rows.append(dict(train_size=int(len(Xtr) * 0.8 * frac), cv_acc=float(np.mean(a)), train_acc=float(np.mean(t))))
        print(name, rows[-1], flush=True)
    out[name] = rows
json.dump(out, open("reports/learning_curve.json", "w"), indent=2)
