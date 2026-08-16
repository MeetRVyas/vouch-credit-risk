import numpy as np

from credit_risk.evaluation.metrics import (
    approval_cutoff_table,
    compute_classification_metrics,
    gini_from_auc,
    ks_statistic,
    reliability_curve,
)


def test_gini_from_auc_perfect_and_random():
    assert gini_from_auc(1.0) == 1.0
    assert gini_from_auc(0.5) == 0.0


def test_compute_classification_metrics_on_perfectly_separable_data():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])

    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics.roc_auc == 1.0
    assert metrics.gini == 1.0
    assert metrics.ks_statistic == 1.0
    assert 0.0 <= metrics.brier_score < 0.05
    assert metrics.pr_auc == 1.0


def test_ks_statistic_zero_when_scores_uninformative():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_prob = np.full(500, 0.5)  # identical score for everyone -> no separation
    assert ks_statistic(y_true, y_prob) == 0.0


def test_reliability_curve_shapes_and_bin_counts_sum_to_n():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(200)

    mean_pred, mean_obs, counts = reliability_curve(y_true, y_prob, n_bins=5)

    assert mean_pred.shape == (5,)
    assert mean_obs.shape == (5,)
    assert counts.sum() == 200


def test_approval_cutoff_table_is_monotonic_in_n_approved():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = rng.random(1000)

    rows = approval_cutoff_table(y_true, y_prob, approval_rates=[0.2, 0.5, 1.0])

    n_approved = [r["n_approved"] for r in rows]
    assert n_approved == sorted(n_approved)
    assert rows[-1]["n_approved"] == 1000
    assert rows[-1]["default_rate_among_approved"] == rows[-1]["overall_default_rate"]
