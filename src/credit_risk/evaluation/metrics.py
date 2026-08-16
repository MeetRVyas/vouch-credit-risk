"""Evaluation metrics for the PD model: ML-generalist metrics plus the
industry-standard credit-scoring ones, plus a business-framing cutoff table.

Kept dependency-light (numpy + sklearn only) so it can run inside an Optuna
objective on every fold without dragging in plotting libraries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    roc_auc: float
    pr_auc: float
    gini: float
    ks_statistic: float
    brier_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "gini": self.gini,
            "ks_statistic": self.ks_statistic,
            "brier_score": self.brier_score,
        }


def gini_from_auc(auc: float) -> float:
    """Gini coefficient, the standard credit-scoring transform of ROC-AUC."""
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    TPR and FPR curves. Standard credit-scoring discrimination metric,
    reported alongside AUC/Gini rather than instead of them."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
    auc = float(roc_auc_score(y_true, y_prob))
    return ClassificationMetrics(
        roc_auc=auc,
        pr_auc=float(average_precision_score(y_true, y_prob)),
        gini=gini_from_auc(auc),
        ks_statistic=ks_statistic(y_true, y_prob),
        brier_score=float(brier_score_loss(y_true, y_prob)),
    )


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bins predictions into equal-width buckets and returns
    (bin_mean_predicted, bin_mean_observed, bin_counts) for a calibration plot."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    mean_predicted = np.zeros(n_bins)
    mean_observed = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            mean_predicted[b] = y_prob[mask].mean()
            mean_observed[b] = y_true[mask].mean()

    return mean_predicted, mean_observed, counts


def approval_cutoff_table(
    y_true: np.ndarray, y_prob: np.ndarray, approval_rates: list[float] | None = None
) -> list[dict]:
    """Business framing: at each approval rate (lowest-risk X% approved),
    report the score cutoff and the resulting default rate among approvals.

    This is the threshold-dependent complement to AUC that the spec calls
    for -- "default rate at different approval-rate cutoffs, not just a
    threshold-free AUC number."
    """
    if approval_rates is None:
        approval_rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    order = np.argsort(y_prob)  # lowest predicted risk first
    sorted_true = y_true[order]
    sorted_prob = y_prob[order]
    n = len(y_true)

    rows = []
    for rate in approval_rates:
        k = max(1, int(round(n * rate)))
        approved_true = sorted_true[:k]
        cutoff_score = sorted_prob[k - 1]
        rows.append(
            {
                "approval_rate": rate,
                "n_approved": int(k),
                "score_cutoff": float(cutoff_score),
                "default_rate_among_approved": float(approved_true.mean()),
                "overall_default_rate": float(y_true.mean()),
            }
        )
    return rows
