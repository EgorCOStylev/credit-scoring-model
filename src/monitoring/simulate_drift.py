import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from src.data.make_dataset import TARGET
from src.features.build_features import RAW_COLUMNS
from src.monitoring.drift import drift_level, psi

MONITORED = ["PAY_0", "LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]


def apply_shift(df):
    df = df.copy()
    df["PAY_0"] = (df["PAY_0"] + 1).clip(upper=8)
    df["LIMIT_BAL"] = df["LIMIT_BAL"] * 0.7
    return df


def score(df, url):
    records = json.loads(df[RAW_COLUMNS].to_json(orient="records"))
    session = requests.Session()
    probs = []
    for record in records:
        resp = session.post(f"{url}/predict", json=record, timeout=10)
        resp.raise_for_status()
        probs.append(resp.json()["default_probability"])
    return pd.Series(probs)


def main(args):
    train = pd.read_csv(args.train).drop(columns=[TARGET])
    test = pd.read_csv(args.test).drop(columns=[TARGET])
    ref = train.sample(args.n, random_state=1).reset_index(drop=True)
    live = test.sample(args.n, random_state=2).reset_index(drop=True)
    if args.shift:
        live = apply_shift(live)

    rows = {}
    for col in MONITORED:
        value = psi(train[col], live[col])
        rows[col] = {"psi": value, "level": drift_level(value)}
    value = psi(score(ref, args.api_url), score(live, args.api_url))
    rows["predicted_probability"] = {"psi": value, "level": drift_level(value)}

    for name, item in rows.items():
        print(f"{name:24s} PSI={item['psi']:.4f}  {item['level']}")
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--report", default="reports/drift_report.json")
    main(parser.parse_args())
