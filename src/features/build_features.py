import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + (
    PAY_COLS + BILL_COLS + PAY_AMT_COLS
)

AGE_BINS = [0, 29, 39, 49, 59, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]

# Codes 0, 5, 6 are undocumented; fold them into "others" (4) / "other" (3)
EDUCATION_MAP = {0: 4, 5: 4, 6: 4}
MARRIAGE_MAP = {0: 3}

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BIN"]
ENGINEERED_NUMERIC = [
    "PAY_MAX",
    "PAY_MEAN",
    "N_DELAYED",
    "BILL_MEAN",
    "UTIL_LAST",
    "UTIL_MEAN",
    "PAY_AMT_MEAN",
    "PAY_RATIO",
    "BILL_TREND",
]
NUMERIC_FEATURES = [
    c for c in RAW_COLUMNS if c not in ["SEX", "EDUCATION", "MARRIAGE"]
] + ENGINEERED_NUMERIC


def normalize_codes(df):
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace(EDUCATION_MAP)
    df["MARRIAGE"] = df["MARRIAGE"].replace(MARRIAGE_MAP)
    return df


def add_features(df):
    out = normalize_codes(df[RAW_COLUMNS])
    pay = out[PAY_COLS]
    bill = out[BILL_COLS]

    out["PAY_MAX"] = pay.max(axis=1)
    out["PAY_MEAN"] = pay.mean(axis=1)
    out["N_DELAYED"] = (pay > 0).sum(axis=1)
    out["BILL_MEAN"] = bill.mean(axis=1)
    out["UTIL_LAST"] = out["BILL_AMT1"] / out["LIMIT_BAL"]
    out["UTIL_MEAN"] = out["BILL_MEAN"] / out["LIMIT_BAL"]
    out["PAY_AMT_MEAN"] = out[PAY_AMT_COLS].mean(axis=1)
    # Share of billed amount actually paid; denominator floored at 1 for zero bills
    out["PAY_RATIO"] = out[PAY_AMT_COLS].sum(axis=1) / np.maximum(
        bill.clip(lower=0).sum(axis=1), 1
    )
    out["BILL_TREND"] = out["BILL_AMT1"] - out["BILL_AMT6"]
    out["AGE_BIN"] = pd.cut(out["AGE"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    # Strings keep the constant imputer and OneHotEncoder happy on integer codes
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        out[col] = out[col].astype(str)
    return out
