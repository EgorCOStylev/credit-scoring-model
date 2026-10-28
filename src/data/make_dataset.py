import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import normalize_codes

TARGET = "default.payment.next.month"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw(path):
    return pd.read_csv(path)


def clean(df):
    # Rows identical apart from ID are mostly inactive accounts (all bills and
    # payments zero), so they are distinct clients and are kept
    df = df.drop(columns=["ID"], errors="ignore")
    return normalize_codes(df).reset_index(drop=True)


def split(df):
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )


def main(raw_path, out_dir):
    from src.data.validation import validate_dataframe

    raw = load_raw(raw_path)
    ok, failures = validate_dataframe(raw)
    if not ok:
        raise SystemExit("Data validation failed:\n" + "\n".join(failures))

    train, test = split(clean(raw))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    main(args.raw_path, args.out_dir)
