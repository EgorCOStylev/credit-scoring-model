import joblib


def load_model(path):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["threshold"]


def predict(pipeline, threshold, df):
    proba = pipeline.predict_proba(df)[:, 1]
    return (proba >= threshold).astype(int), proba
