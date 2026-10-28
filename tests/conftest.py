from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_df():
    return pd.read_csv(DATA_DIR / "sample.csv")
