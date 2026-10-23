"""Data loading and feature engineering for the Telco churn dataset."""
import numpy as np
import pandas as pd

DATA_PATH = "data/customer_churn.csv"
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # The 11 blank TotalCharges rows are brand-new customers (tenure == 0).
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])
    return X, y


def add_features(X):
    """Engineered features; only uses fields available at prediction time."""
    X = X.copy()
    X["NumAddons"] = sum((X[c] == "Yes").astype(int) for c in ADDON_COLS)
    X["AvgCharge"] = np.where(X["tenure"] > 0, X["TotalCharges"] / X["tenure"].clip(lower=1), X["MonthlyCharges"])
    X["ChargeDrift"] = X["MonthlyCharges"] - X["AvgCharge"]
    X["ExpectedTotal"] = X["MonthlyCharges"] * X["tenure"]
    X["TotalResidual"] = X["TotalCharges"] - X["ExpectedTotal"]
    X["IsNewCustomer"] = (X["tenure"] <= 6).astype(int)
    X["AutoPay"] = X["PaymentMethod"].str.contains("automatic").astype(int)
    X["FiberNoSupport"] = ((X["InternetService"] == "Fiber optic") & (X["TechSupport"] != "Yes")).astype(int)
    X["M2M_Electronic"] = ((X["Contract"] == "Month-to-month") & (X["PaymentMethod"] == "Electronic check")).astype(int)
    X["Family"] = ((X["Partner"] == "Yes") | (X["Dependents"] == "Yes")).astype(int)
    return X
