import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from scipy.stats import loguniform  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from src.data.make_dataset import TARGET  # noqa: E402
from src.models.evaluate import (  # noqa: E402
    choose_threshold,
    compute_metrics,
    plot_roc,
)
from src.models.pipeline import RANDOM_STATE, create_pipeline  # noqa: E402

EXPERIMENT_NAME = "Credit_Default_Prediction"
CV_FOLDS = 5

EXPERIMENTS = [
    {
        "name": "logreg_l2",
        "model": "logreg",
        "n_iter": 8,
        "params": {
            "classifier__C": loguniform(1e-3, 1e1),
            "classifier__class_weight": [None, "balanced"],
        },
    },
    {
        "name": "rf_shallow",
        "model": "rf",
        "n_iter": 8,
        "params": {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [4, 6, 8],
            "classifier__min_samples_leaf": [20, 50],
            "classifier__class_weight": [None, "balanced_subsample"],
        },
    },
    {
        "name": "rf_deep",
        "model": "rf",
        "n_iter": 6,
        "params": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [12, 16, None],
            "classifier__min_samples_leaf": [5, 10],
            "classifier__max_features": ["sqrt", 0.5],
        },
    },
    {
        "name": "gb_base",
        "model": "gb",
        "n_iter": 8,
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
            "classifier__subsample": [0.8, 1.0],
        },
    },
    {
        "name": "hgb",
        "model": "hgb",
        "n_iter": 8,
        "params": {
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_iter": [200, 400],
            "classifier__max_depth": [3, 5, None],
            "classifier__l2_regularization": [0.0, 1.0, 10.0],
            "classifier__min_samples_leaf": [20, 50],
        },
    },
    {
        "name": "hgb_regularized",
        "model": "hgb",
        "n_iter": 6,
        "params": {
            "classifier__learning_rate": [0.02, 0.03],
            "classifier__max_iter": [300, 600],
            "classifier__max_depth": [3, 4],
            "classifier__l2_regularization": [10.0, 30.0],
            "classifier__min_samples_leaf": [50, 100],
        },
    },
]


def run_experiment(exp, X_train, y_train, X_test, y_test, n_iter=None):
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_iter = n_iter or exp["n_iter"]
    search = RandomizedSearchCV(
        create_pipeline(exp["model"]),
        exp["params"],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    with mlflow.start_run(run_name=exp["name"]):
        mlflow.log_params(
            {"model_type": exp["model"], "n_iter": n_iter, "cv_folds": CV_FOLDS}
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        mlflow.log_metric("cv_roc_auc", float(search.best_score_))

        # Threshold comes from out-of-fold train predictions, never from test
        oof = cross_val_predict(
            best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        threshold = choose_threshold(y_train, oof)
        proba = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba, threshold)

        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        with tempfile.TemporaryDirectory() as tmp:
            roc_path = Path(tmp) / "roc_curve.png"
            plot_roc(y_test, proba, roc_path, title=exp["name"])
            mlflow.log_artifact(str(roc_path))
        mlflow.sklearn.log_model(
            best,
            name="model",
            signature=infer_signature(X_test.head(5), proba[:5]),
            serialization_format="cloudpickle",
        )

    return {
        "name": exp["name"],
        "cv_roc_auc": float(search.best_score_),
        "threshold": threshold,
        "metrics": metrics,
        "pipeline": best,
        "proba": proba,
    }


def main(train_path, test_path, models_dir, reports_dir, metrics_path, n_iter, only):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    experiments = [e for e in EXPERIMENTS if not only or e["name"] in only]
    results = []
    for exp in experiments:
        res = run_experiment(exp, X_train, y_train, X_test, y_test, n_iter)
        results.append(res)
        print(
            f"{res['name']}: cv_auc={res['cv_roc_auc']:.4f} "
            f"test_auc={res['metrics']['roc_auc']:.4f} "
            f"f1={res['metrics']['f1']:.4f}"
        )

    # Best model is picked by cross-validated AUC on train, not by test score
    best = max(results, key=lambda r: r["cv_roc_auc"])
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best["pipeline"], "threshold": best["threshold"]},
        models_dir / "credit_default_model.pkl",
    )
    plot_roc(y_test, best["proba"], reports_dir / "roc_curve.png", best["name"])
    summary = {
        "model": best["name"],
        "cv_roc_auc": best["cv_roc_auc"],
        "threshold": best["threshold"],
        **{f"test_{k}": v for k, v in best["metrics"].items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("best:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--metrics", default="metrics.json")
    parser.add_argument("--n-iter", type=int)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    main(
        args.train,
        args.test,
        args.models_dir,
        args.reports_dir,
        args.metrics,
        args.n_iter,
        args.only,
    )
