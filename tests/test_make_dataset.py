from src.data.make_dataset import TARGET, clean, split


def test_clean_drops_id_and_folds_codes(sample_df):
    df = sample_df.copy()
    df.loc[df.index[0], "EDUCATION"] = 5
    df.loc[df.index[1], "EDUCATION"] = 0
    df.loc[df.index[2], "MARRIAGE"] = 0

    out = clean(df)

    assert "ID" not in out.columns
    assert len(out) == len(df)
    assert set(out["EDUCATION"]) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"]) <= {1, 2, 3}


def test_split_is_stratified_and_disjoint(sample_df):
    df = clean(sample_df)
    train, test = split(df)

    assert len(train) + len(test) == len(df)
    assert len(test) == int(len(df) * 0.2)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.02
    assert not set(train.index) & set(test.index)
