import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import load_model, predict

MODEL_PATH = os.environ.get("MODEL_PATH", "models/credit_default_model.pkl")

app = FastAPI(title="Credit Default Prediction API")
_bundle = None


class ClientData(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_0: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_model(MODEL_PATH)
    return _bundle


@app.get("/")
def read_root():
    return {"message": "Credit Default Prediction API is alive"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_default(data: ClientData):
    try:
        pipeline, threshold = get_bundle()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model file not found")
    df = pd.DataFrame([data.model_dump()])
    pred, proba = predict(pipeline, threshold, df)
    return {
        "default_prediction": int(pred[0]),
        "default_probability": float(proba[0]),
    }
