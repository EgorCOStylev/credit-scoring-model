from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
)

RANDOM_STATE = 42


def make_classifier(name):
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    if name == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


def create_pipeline(model_name):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocessor", preprocessor),
            ("classifier", make_classifier(model_name)),
        ]
    )
