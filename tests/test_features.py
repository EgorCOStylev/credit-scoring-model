import numpy as np
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    add_features,
)


def make_row(**overrides):
    row = {c: 0 for c in RAW_COLUMNS}
    row.update(LIMIT_BAL=100000, SEX=1, EDUCATION=2, MARRIAGE=1, AGE=35)
    row.update(overrides)
    return pd.DataFrame([row])


def test_output_columns_match_feature_lists(sample_df):
    out = add_features(sample_df)
    assert set(out.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert not out[NUMERIC_FEATURES].isna().any().any()
    assert np.isfinite(out[NUMERIC_FEATURES]).all().all()


def test_payment_history_aggregates():
    row = make_row(PAY_0=2, PAY_2=1, PAY_3=-1, PAY_4=0, PAY_5=0, PAY_6=-2)
    out = add_features(row).iloc[0]
    assert out["PAY_MAX"] == 2
    assert out["N_DELAYED"] == 2
    assert out["PAY_MEAN"] == 0


def test_utilization_and_pay_ratio():
    row = make_row(BILL_AMT1=50000, BILL_AMT2=50000, PAY_AMT1=25000)
    out = add_features(row).iloc[0]
    assert out["UTIL_LAST"] == 0.5
    assert out["PAY_RATIO"] == 0.25


def test_pay_ratio_with_zero_bills_is_finite():
    out = add_features(make_row(PAY_AMT1=1000)).iloc[0]
    assert np.isfinite(out["PAY_RATIO"])


def test_age_bins_on_boundaries():
    ages = [21, 29, 30, 39, 40, 59, 60, 79]
    df = pd.concat([make_row(AGE=a) for a in ages], ignore_index=True)
    bins = add_features(df)["AGE_BIN"].tolist()
    assert bins == ["<30", "<30", "30-39", "30-39", "40-49", "50-59", "60+", "60+"]


def test_undocumented_codes_are_folded():
    out = add_features(make_row(EDUCATION=5, MARRIAGE=0)).iloc[0]
    assert out["EDUCATION"] == "4"
    assert out["MARRIAGE"] == "3"
