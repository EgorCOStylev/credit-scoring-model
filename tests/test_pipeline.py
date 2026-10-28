import pytest

from src.data.make_dataset import TARGET
from src.models.pipeline import create_pipeline


@pytest.mark.parametrize("model", ["logreg", "rf", "gb", "hgb"])
def test_pipeline_fits_and_predicts_probabilities(sample_df, model):
    X = sample_df.drop(columns=[TARGET])
    y = sample_df[TARGET]
    pipe = create_pipeline(model).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_accepts_raw_codes_unseen_in_training(sample_df):
    X = sample_df.drop(columns=[TARGET])
    y = sample_df[TARGET]
    pipe = create_pipeline("logreg").fit(X, y)
    row = X.head(1).copy()
    row["EDUCATION"] = 6
    row["MARRIAGE"] = 0
    assert pipe.predict_proba(row).shape == (1, 2)
