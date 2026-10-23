"""Model/pipeline factory shared by experiments and final training."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import add_features


def preprocessor(X, scale=True):
    cat = list(X.select_dtypes(include=["object", "string"]).columns)
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler() if scale else "passthrough", num),
    ])


def make_pipeline(name, X_sample, engineered=True, seed=42):
    Xs = add_features(X_sample) if engineered else X_sample
    fe = FunctionTransformer(add_features) if engineered else FunctionTransformer()

    def wrap(clf, scale=True):
        return Pipeline([("fe", fe), ("prep", preprocessor(Xs, scale)), ("clf", clf)])

    lr = lambda C=0.1: LogisticRegression(C=C, max_iter=5000)
    hgb = lambda: HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_depth=3,
                                                 min_samples_leaf=40, l2_regularization=5.0, random_state=seed)
    xgb = lambda: XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                                colsample_bytree=0.6, min_child_weight=5, reg_lambda=5.0,
                                random_state=seed, n_jobs=4, eval_metric="logloss")
    gb = lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.04, max_depth=3,
                                            subsample=0.8, random_state=seed)
    rf = lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features=0.3,
                                        n_jobs=4, random_state=seed)
    if name == "logreg":
        return wrap(lr(1.0))
    if name == "logreg_C0.1":
        return wrap(lr(0.1))
    if name == "logreg_C0.02":
        return wrap(lr(0.02))
    if name == "hgb":
        return wrap(hgb(), scale=False)
    if name == "xgb":
        return wrap(xgb(), scale=False)
    if name == "gb":
        return wrap(gb(), scale=False)
    if name == "rf":
        return wrap(rf(), scale=False)
    if name == "soft_vote":
        return wrap(VotingClassifier([("lr", lr(0.1)), ("xgb", xgb()), ("hgb", hgb())], voting="soft"))
    if name == "stack":
        return wrap(StackingClassifier(
            [("lr", lr(0.1)), ("xgb", xgb()), ("rf", rf())],
            final_estimator=LogisticRegression(C=1.0), cv=5, n_jobs=1))
    raise ValueError(name)
