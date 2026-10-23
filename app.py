import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.features import load_data  # noqa: F401  (also needed so joblib can unpickle src.* functions)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def get_bundle():
    return joblib.load("model.pkl")


@st.cache_data
def get_data():
    X, y = load_data()
    return X, y


bundle = get_bundle()
pipe, THR, M = bundle["pipeline"], bundle["threshold"], bundle["metrics"]
X, y = get_data()
COLS = bundle["columns"]

CHOICES = {
    "gender": ["Female", "Male"], "Partner": ["No", "Yes"], "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"], "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    **{c: ["No", "Yes", "No internet service"] for c in
       ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]},
    "Contract": ["Month-to-month", "One year", "Two year"], "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
PRESETS = {
    "High-risk: new fiber, month-to-month": dict(gender="Female", SeniorCitizen=0, Partner="No", Dependents="No", tenure=2,
        PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="No", StreamingMovies="No", Contract="Month-to-month",
        PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=75.0, TotalCharges=150.0),
    "Low-risk: loyal two-year DSL": dict(gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=60,
        PhoneService="Yes", MultipleLines="Yes", InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
        DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No", StreamingMovies="No", Contract="Two year",
        PaperlessBilling="No", PaymentMethod="Credit card (automatic)", MonthlyCharges=65.0, TotalCharges=3900.0),
    "Borderline: mid-tenure one-year": dict(gender="Male", SeniorCitizen=1, Partner="No", Dependents="No", tenure=20,
        PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="Yes",
        DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes", Contract="One year",
        PaperlessBilling="Yes", PaymentMethod="Mailed check", MonthlyCharges=90.0, TotalCharges=1800.0),
}


def tier(p):
    if p >= 0.6:
        return "High", "error"
    if p >= THR:
        return "Medium", "warning"
    return "Low", "success"


def explain(row: pd.DataFrame, base_p: float):
    """Model-agnostic what-if: swap each field for the typical customer's value, measure probability change."""
    ref = {c: (X[c].mode()[0] if X[c].dtype == object or str(X[c].dtype).startswith("str") else X[c].median()) for c in COLS}
    variants = []
    for c in COLS:
        v = row.copy(); v[c] = ref[c]; variants.append(v)
    ps = pipe.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    return pd.Series(base_p - ps, index=COLS)  # + => this field pushes risk up vs. a typical customer


st.sidebar.title("📉 Churn Predictor")
page = st.sidebar.radio("Navigate", ["Predict a customer", "Batch scoring", "Data overview", "Model & honesty report"])
st.sidebar.caption(f"Model: soft-voting ensemble (logistic regression + XGBoost + HistGradientBoosting)\n\n"
                   f"Hold-out accuracy **{M['at_0_5']['accuracy']:.1%}**, ROC-AUC **{M['roc_auc']:.3f}**")

# ---------------------------------------------------------------- Predict
if page == "Predict a customer":
    st.title("Predict a customer's churn risk")
    preset = st.selectbox("Start from an example", ["(custom)"] + list(PRESETS))
    d = PRESETS.get(preset, PRESETS["Borderline: mid-tenure one-year"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, int(d["tenure"]))
        contract = st.selectbox("Contract", CHOICES["Contract"], CHOICES["Contract"].index(d["Contract"]))
        paperless = st.selectbox("Paperless billing", CHOICES["PaperlessBilling"], CHOICES["PaperlessBilling"].index(d["PaperlessBilling"]))
        payment = st.selectbox("Payment method", CHOICES["PaymentMethod"], CHOICES["PaymentMethod"].index(d["PaymentMethod"]))
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, float(d["MonthlyCharges"]), 0.5)
        auto_total = round(monthly * tenure, 2)
        total = st.number_input("Total charges ($)", 0.0, 10000.0, float(d["TotalCharges"]), 1.0,
                                help=f"Roughly monthly × tenure = ${auto_total:,.0f}")
    with c2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone service", CHOICES["PhoneService"], CHOICES["PhoneService"].index(d["PhoneService"]))
        multiple = st.selectbox("Multiple lines", CHOICES["MultipleLines"], CHOICES["MultipleLines"].index(d["MultipleLines"]))
        internet = st.selectbox("Internet service", CHOICES["InternetService"], CHOICES["InternetService"].index(d["InternetService"]))
        addons = {c: st.selectbox(c, CHOICES[c], CHOICES[c].index(d[c])) for c in
                  ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]}
    with c3:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", CHOICES["gender"], CHOICES["gender"].index(d["gender"]))
        senior = st.selectbox("Senior citizen", [0, 1], int(d["SeniorCitizen"]), format_func=lambda v: "Yes" if v else "No")
        partner = st.selectbox("Partner", CHOICES["Partner"], CHOICES["Partner"].index(d["Partner"]))
        deps = st.selectbox("Dependents", CHOICES["Dependents"], CHOICES["Dependents"].index(d["Dependents"]))

    # keep inputs logically consistent with the dataset's conventions
    if phone == "No":
        multiple = "No phone service"
    if internet == "No":
        addons = {k: "No internet service" for k in addons}
    elif any(v == "No internet service" for v in addons.values()):
        addons = {k: ("No" if v == "No internet service" else v) for k, v in addons.items()}

    row = pd.DataFrame([dict(gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=deps, tenure=tenure,
                             PhoneService=phone, MultipleLines=multiple, InternetService=internet, **addons,
                             Contract=contract, PaperlessBilling=paperless, PaymentMethod=payment,
                             MonthlyCharges=monthly, TotalCharges=total)])[COLS]
    prob = float(pipe.predict_proba(row)[0, 1])
    name, kind = tier(prob)
    st.divider()
    a, b = st.columns([1, 2])
    with a:
        st.metric("Churn probability", f"{prob:.1%}", help="Calibrated-ish estimate from the ensemble")
        getattr(st, kind)(f"**{name} risk** — decision threshold {THR:.0%} (tuned for F1; flags {'CHURN' if prob >= THR else 'STAY'})")
        st.progress(min(prob, 1.0))
        st.caption(f"Base rate in the data: {y.mean():.1%}. Risk tiers: Low < {THR:.0%} ≤ Medium < 60% ≤ High.")
    with b:
        st.markdown("**Why? Biggest drivers for this customer**")
        st.caption("Each bar = how much the churn probability changes versus a typical customer's value for that field.")
        eff = explain(row, prob)
        top = eff.reindex(eff.abs().sort_values(ascending=False).index)[:8][::-1]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(top.index, top.values * 100, color=["#d1495b" if v > 0 else "#2e86ab" for v in top.values])
        ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("Change in churn probability (percentage points)")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        st.pyplot(fig, clear_figure=True)
    if name != "Low":
        st.info("**Retention ideas:** " + ("offer a 1–2 year contract incentive; " if contract == "Month-to-month" else "")
                + ("bundle Tech Support / Online Security; " if internet != "No" and addons.get("TechSupport") != "Yes" else "")
                + ("move to automatic payment; " if "automatic" not in payment else "")
                + "prioritise outreach in the first 6 months.")

# ---------------------------------------------------------------- Batch
elif page == "Batch scoring":
    st.title("Score a list of customers")
    st.write("Upload a CSV with the original dataset's columns (`customerID` optional, `Churn` ignored).")
    f = st.file_uploader("CSV file", type="csv")
    if st.button("Use the sample dataset instead"):
        f = "data/customer_churn.csv"
    if f is not None:
        raw = pd.read_csv(f)
        raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
        missing = [c for c in COLS if c not in raw]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            out = raw.copy()
            out["churn_probability"] = pipe.predict_proba(raw[COLS])[:, 1]
            out["risk_tier"] = out["churn_probability"].map(lambda p: tier(p)[0])
            c1, c2, c3 = st.columns(3)
            c1.metric("Customers", f"{len(out):,}"); c2.metric("Flagged as churn", f"{(out.churn_probability >= THR).sum():,}")
            c3.metric("High risk", f"{(out.risk_tier == 'High').sum():,}")
            st.dataframe(out.sort_values("churn_probability", ascending=False).head(200), use_container_width=True)
            st.download_button("Download scored CSV", out.to_csv(index=False), "scored_customers.csv", "text/csv")
            if f == "data/customer_churn.csv":
                st.caption("⚠️ The final model was refit on this same data, so scores here are in-sample, not an accuracy test.")

# ---------------------------------------------------------------- Overview
elif page == "Data overview":
    st.title("Data overview")
    df = X.assign(Churn=y)
    a, b, c, d_ = st.columns(4)
    a.metric("Customers", f"{len(df):,}"); b.metric("Churned", f"{y.sum():,}"); c.metric("Churn rate", f"{y.mean():.1%}")
    d_.metric("Avg monthly charge", f"${df.MonthlyCharges.mean():.0f}")
    l, r = st.columns(2)
    with l:
        st.markdown("**Churn rate by contract**"); st.bar_chart(df.groupby("Contract").Churn.mean() * 100)
        st.markdown("**Churn rate by internet service**"); st.bar_chart(df.groupby("InternetService").Churn.mean() * 100)
    with r:
        st.markdown("**Churn rate by tenure band**")
        band = pd.cut(df.tenure, [-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"])
        st.bar_chart(df.groupby(band, observed=True).Churn.mean() * 100)
        st.markdown("**Churn rate by payment method**"); st.bar_chart(df.groupby("PaymentMethod").Churn.mean() * 100)

# ---------------------------------------------------------------- Model report
else:
    st.title("Model & honesty report")
    h = M["at_0_5"]; t = M["at_tuned"]
    st.success(f"Held-out accuracy **{h['accuracy']:.1%}** (95% CI {M['accuracy_ci95'][0]:.1%}–{M['accuracy_ci95'][1]:.1%}) "
               f"vs. {M['majority_baseline_accuracy']:.1%} for always predicting “no churn”. ROC-AUC **{M['roc_auc']:.3f}**.")
    st.markdown("""
**Why not 90–95 %?** Extensive testing shows ~80–81 % is the ceiling for this dataset (7,043 customers, 19 fields):
- Ten model configurations (logistic regression, random forest, gradient boosting, XGBoost, HistGB, voting and stacking ensembles, with and without engineered features) all land at **80.3–80.8 %** in repeated 5×3 cross-validation (AUC 0.84–0.85). Differences are far smaller than the fold-to-fold noise (±0.8 pp).
- For logistic regression, **train accuracy ≈ CV accuracy** and the learning curve is flat — more data or a bigger model will not help; the missing information is not in the columns (no usage, complaints, support calls, price changes, competitor offers).
- Any 90 %+ result on this data would indicate leakage or evaluation on training rows, so none is claimed.

What *did* improve: probability quality (log-loss 0.417 → 0.411) and a tuned decision threshold for catching more churners.
""")
    c1, c2 = st.columns(2)
    tbl = pd.DataFrame({"Threshold 0.50": h, f"Tuned ({t['threshold']:.2f})": t}).loc[["accuracy", "precision", "recall", "f1"]]
    c1.markdown("**Hold-out metrics (1,409 unseen customers)**"); c1.dataframe(tbl.style.format("{:.1%}"))
    c1.caption("Lower threshold ⇒ catches more churners (recall ↑) at the cost of more false alarms and lower accuracy. "
               "Choose by the cost of a lost customer vs. a wasted retention offer.")
    fig, ax = plt.subplots(figsize=(4, 3))
    cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix @ {t['threshold']:.2f}")
    c2.pyplot(fig, clear_figure=True)
    st.markdown("**Global feature importance** (permutation, drop in hold-out AUC)")
    imp = pd.Series(M["permutation_importance"]).head(10)[::-1]
    st.bar_chart(imp)
    cvp = Path("reports/cv_results.json")
    if cvp.exists():
        st.markdown("**Cross-validated model comparison** (development split, 5-fold × 3 repeats)")
        r = pd.DataFrame(json.load(open(cvp)))[["model", "engineered", "acc", "acc_sd", "auc", "f1", "logloss"]]
        st.dataframe(r.style.format({"acc": "{:.2%}", "acc_sd": "{:.2%}", "auc": "{:.3f}", "f1": "{:.3f}", "logloss": "{:.3f}"}),
                     hide_index=True)
