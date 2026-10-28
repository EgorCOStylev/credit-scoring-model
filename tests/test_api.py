import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import app as app_module  # noqa: E402
from src.data.make_dataset import TARGET  # noqa: E402
from src.models.pipeline import create_pipeline  # noqa: E402

EXAMPLE = Path(__file__).parent / "data" / "example_request.json"


@pytest.fixture
def client(sample_df):
    pipe = create_pipeline("logreg").fit(
        sample_df.drop(columns=[TARGET]), sample_df[TARGET]
    )
    app_module._bundle = (pipe, 0.5)
    yield TestClient(app_module.app)
    app_module._bundle = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_class_and_probability(client):
    payload = json.loads(EXAMPLE.read_text())
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_prediction"] in (0, 1)
    assert 0 <= body["default_probability"] <= 1


def test_predict_rejects_invalid_input(client):
    payload = json.loads(EXAMPLE.read_text())
    payload["SEX"] = 5
    assert client.post("/predict", json=payload).status_code == 422
