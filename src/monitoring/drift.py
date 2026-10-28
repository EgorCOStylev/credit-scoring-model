import numpy as np

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def psi(expected, actual, bins=10, eps=1e-6):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    values = np.unique(expected)

    if len(values) <= bins:
        # Discrete feature: one bin per value, plus one for values unseen in train
        exp_counts = np.array([(expected == v).sum() for v in values] + [0])
        act_counts = np.array(
            [(actual == v).sum() for v in values] + [(~np.isin(actual, values)).sum()]
        )
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        inner = edges[1:-1]
        exp_counts = np.bincount(
            np.searchsorted(inner, expected, side="right"), minlength=len(inner) + 1
        )
        act_counts = np.bincount(
            np.searchsorted(inner, actual, side="right"), minlength=len(inner) + 1
        )

    exp_share = np.clip(exp_counts / exp_counts.sum(), eps, None)
    act_share = np.clip(act_counts / act_counts.sum(), eps, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def drift_level(value):
    if value < PSI_MODERATE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"
