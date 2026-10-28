import numpy as np

from src.monitoring.drift import drift_level, psi


def test_psi_of_identical_samples_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-9


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert psi(a, b) < 0.1


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(loc=1.0, size=5000)
    assert psi(a, b) > 0.25


def test_psi_discrete_feature_with_unseen_value():
    expected = np.array([0] * 50 + [1] * 50)
    actual = np.array([0] * 50 + [2] * 50)
    assert psi(expected, actual) > 0.25


def test_drift_level_thresholds():
    assert drift_level(0.05) == "stable"
    assert drift_level(0.15) == "moderate"
    assert drift_level(0.3) == "significant"
