"""SHAP explainability: global summary plot + per-prediction contributions.

Credit models need adverse-action-style explainability (which features
pushed *this* applicant's score up or down), not just a global importance
plot -- so this module exposes both.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: notebooks/servers, no display backend required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def build_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    return explainer.shap_values(X)


def save_summary_plot(shap_values: np.ndarray, X: pd.DataFrame, out_path: str, max_display: int = 20) -> str:
    """Global SHAP summary (beeswarm) plot -- the deliverable in spec sec. 8."""
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def top_contributors_for_row(
    shap_values: np.ndarray, X: pd.DataFrame, row_idx: int, top_n: int = 5
) -> list[dict]:
    """Adverse-action-style explanation for a single prediction: the top
    features pushing that applicant's score up or down, with signed
    contribution and the feature's raw value."""
    row_shap = shap_values[row_idx]
    row_values = X.iloc[row_idx]
    order = np.argsort(-np.abs(row_shap))[:top_n]

    return [
        {
            "feature": X.columns[i],
            "value": row_values.iloc[i],
            "shap_contribution": float(row_shap[i]),
            "direction": "increases_risk" if row_shap[i] > 0 else "decreases_risk",
        }
        for i in order
    ]
