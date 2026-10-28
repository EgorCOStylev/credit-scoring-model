import numpy as np
import pytest

pytest.importorskip("great_expectations")

from src.data.validation import validate_dataframe  # noqa: E402

TARGET = "default.payment.next.month"


def set_first(df, col, value):
    df[col] = df[col].astype(float)
    df.loc[df.index[0], col] = value


def make_null(df):
    set_first(df, "LIMIT_BAL", np.nan)


def make_age_out_of_range(df):
    set_first(df, "AGE", 5)


def make_sex_not_in_set(df):
    set_first(df, "SEX", 3)


def make_pay_out_of_range(df):
    set_first(df, "PAY_0", 42)


def make_negative_payment(df):
    set_first(df, "PAY_AMT1", -10)


def make_bad_target(df):
    df[TARGET] = 2


def make_wrong_type(df):
    df["EDUCATION"] = df["EDUCATION"].astype(str)


def make_missing_column(df):
    del df["BILL_AMT3"]


def make_duplicate_id(df):
    df.loc[df.index[1], "ID"] = df.loc[df.index[0], "ID"]


ANOMALIES = [
    make_null,
    make_age_out_of_range,
    make_sex_not_in_set,
    make_pay_out_of_range,
    make_negative_payment,
    make_bad_target,
    make_wrong_type,
    make_missing_column,
    make_duplicate_id,
]


def test_clean_sample_passes(sample_df):
    ok, failures = validate_dataframe(sample_df)
    assert ok, failures


@pytest.mark.parametrize("mutate", ANOMALIES, ids=lambda f: f.__name__)
def test_anomalies_fail_validation(sample_df, mutate):
    df = sample_df.copy()
    mutate(df)
    ok, failures = validate_dataframe(df)
    assert not ok
    assert failures
