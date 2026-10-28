import numpy as np

from src.models.evaluate import choose_threshold, compute_metrics, plot_roc


def test_metrics_on_known_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    m = compute_metrics(y, proba, threshold=0.5)
    # tp=2 fp=1 fn=1
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert abs(m["f1"] - 2 / 3) < 1e-12
    assert 0.5 < m["roc_auc"] <= 1


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, proba, threshold=0.5)
    assert m == {"roc_auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_choose_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    threshold = choose_threshold(y, proba)
    assert 0.3 < threshold <= 0.6


def test_plot_roc_writes_file(tmp_path):
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.35, 0.8])
    path = tmp_path / "roc.png"
    plot_roc(y, proba, path)
    assert path.stat().st_size > 0
