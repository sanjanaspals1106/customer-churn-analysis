# Customer Churn Prediction (Telco)

Predicts whether a telecom customer will churn, with a Streamlit app for single-customer prediction,
batch scoring, data exploration and a transparent model report.

## Results (honest, reproducible)
| | |
|---|---|
| Hold-out accuracy (1,409 unseen customers, threshold 0.5) | **80.5 %** (95 % CI 78.4–82.5 %) |
| Majority-class baseline | 73.5 % |
| ROC-AUC / PR-AUC | 0.849 / 0.648 |
| Churn recall / precision @ 0.5 | 54 % / 66 % |
| Churn recall / precision @ tuned threshold 0.34 | 72 % / 56 % (accuracy 77.4 %) |

**Why not 90–95 %?** On this dataset it is not achievable under valid evaluation:
- Ten configurations (logistic regression, RF, GB, HistGB, XGBoost, voting and stacking ensembles, ± engineered features)
  all score **80.3–80.8 %** in 5-fold × 3-repeat CV (AUC 0.84–0.85); gaps are smaller than the ±0.8 pp fold noise.
- Logistic regression's train accuracy ≈ CV accuracy and the learning curve is flat (`reports/learning_curve.json`):
  the limit is information, not model capacity or data volume. The data lacks usage, complaints, support calls, pricing changes.
- Duplicate feature rows with conflicting labels exist, confirming label noise.
- Anything ≥ 90 % here would imply target leakage or scoring on training data. Feature engineering, tuning and ensembling
  gave only improved probability quality (log-loss 0.417 → 0.411), not accuracy.

The best defensible deliverable is therefore a well-calibrated ranking model (AUC ≈ 0.85) with a tunable threshold:
lowering it trades accuracy for catching more churners, which is usually the right business choice.

## Protocol
20 % stratified hold-out (seed 2024) is evaluated once; all model selection uses CV on the other 80 %; the F1 threshold is chosen
on out-of-fold predictions only. The shipped `model.pkl` is refit on all rows after evaluation.
Note: `customerID` is dropped; the 11 blank `TotalCharges` rows (tenure 0) are set to 0 rather than discarded.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                         # retrain -> model.pkl, reports/metrics.json (~1 min)
streamlit run app.py                    # open http://localhost:8501
python experiments/run_experiments.py   # optional: full CV comparison (~5 min) -> reports/cv_results.json
python experiments/diagnostics.py       # optional: learning curves
```
Layout: `src/features.py` (loading + features), `src/models.py` (pipelines), `train.py`, `app.py`, `analysis.ipynb` (original EDA, kept).

## App pages
Predict a customer (presets, probability, risk tier, per-customer driver chart, retention ideas) · Batch scoring (CSV in/out) ·
Data overview · Model & honesty report.
