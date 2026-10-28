import argparse
import json
import sys

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from src.features.build_features import RAW_COLUMNS

TARGET = "default.payment.next.month"
SUITE_NAME = "credit_data_suite"
COLUMNS = ["ID"] + RAW_COLUMNS + [TARGET]
NUMERIC_TYPES = ["int64", "float64"]

# Bounds are wider than the observed data (LIMIT_BAL 1e4..1e6, AGE 21..79,
# BILL_AMT -3.4e5..1.7e6, PAY_AMT 0..1.7e6) so only real anomalies trip them
RANGES = {
    "LIMIT_BAL": (1, 2_000_000),
    "AGE": (18, 100),
    **{f"PAY_{i}": (-2, 9) for i in [0, 2, 3, 4, 5, 6]},
    **{f"BILL_AMT{i}": (-1_000_000, 3_000_000) for i in range(1, 7)},
    **{f"PAY_AMT{i}": (0, 3_000_000) for i in range(1, 7)},
}
VALUE_SETS = {
    "SEX": [1, 2],
    "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
    "MARRIAGE": [0, 1, 2, 3],
    TARGET: [0, 1],
}


def build_suite():
    suite = gx.ExpectationSuite(name=SUITE_NAME)
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=100))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="ID"))
    for col in COLUMNS:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInTypeList(column=col, type_list=NUMERIC_TYPES)
        )
    for col, (low, high) in RANGES.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=low, max_value=high)
        )
    for col, values in VALUE_SETS.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=col, value_set=values)
        )
    return suite


def validate_dataframe(df):
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("credit")
    asset = source.add_dataframe_asset("raw")
    batch_def = asset.add_batch_definition_whole_dataframe("whole")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(build_suite())
    failures = []
    for item in result.results:
        if not item.success:
            cfg = item.expectation_config
            failures.append(f"{cfg.type}: {cfg.kwargs}")
    return bool(result.success), failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--dump-suite")
    args = parser.parse_args()

    if args.dump_suite:
        with open(args.dump_suite, "w", encoding="utf-8") as f:
            json.dump(build_suite().to_json_dict(), f, indent=2, ensure_ascii=False)

    ok, failures = validate_dataframe(pd.read_csv(args.path))
    if not ok:
        print(f"Validation failed ({len(failures)} expectations):")
        print("\n".join(failures))
        sys.exit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
