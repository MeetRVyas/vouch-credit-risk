"""Stratified-CV XGBoost training with Optuna tuning, logged to MLflow.

The pandas conversion happens only at the very last step, right before
`.fit()` -- everything upstream (DuckDB join, Polars features) stays in
Polars/DuckDB. That's the "thin conversion only" boundary from the spec.

On Kaggle, pass `device="cuda"` via `XGBParams(device="cuda")`; locally
(no GPU in this sandbox) the default is "cpu". `tree_method="hist"` is used
either way, per the spec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold

from credit_risk.evaluation.metrics import compute_classification_metrics

load_dotenv()

DEVICE = os.environ.get("DEVICE", "cpu")

@dataclass
class XGBParams:
    tree_method: str = "hist"
    device: str = DEVICE  # set to "cuda" when running on a Kaggle GPU session
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0

    def to_xgb_kwargs(self, scale_pos_weight: float) -> dict:
        return {
            "tree_method": self.tree_method,
            "device": self.device,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": -1,
        }


@dataclass
class CVResult:
    fold_metrics: list[dict] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None
    mean_metrics: dict = field(default_factory=dict)


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """neg/pos ratio -- standard scale_pos_weight for XGBoost class imbalance."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / max(n_pos, 1))


def run_stratified_cv(
    X: pl.DataFrame,
    y: np.ndarray,
    params: XGBParams,
    n_splits: int = 5,
    seed: int = 42,
) -> CVResult:
    """Stratified K-fold CV (never a single split -- target is ~8% positive)."""
    X_pd = X.to_pandas()  # thin conversion, at the fit boundary only
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pd, y)):
        X_train, X_val = X_pd.iloc[train_idx], X_pd.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        spw = compute_scale_pos_weight(y_train)
        model = xgb.XGBClassifier(**params.to_xgb_kwargs(spw))
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_prob = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob

        metrics = compute_classification_metrics(y_val, val_prob).as_dict()
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    mean_metrics = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in fold_metrics[0]
        if k != "fold"
    }

    return CVResult(fold_metrics=fold_metrics, oof_predictions=oof, mean_metrics=mean_metrics)


def make_optuna_objective(X: pl.DataFrame, y: np.ndarray, n_splits: int = 5, device: str = DEVICE):
    """Optuna objective maximizing mean CV ROC-AUC. Bayesian search
    (TPESampler, Optuna's default) under a limited compute budget, per spec."""

    def objective(trial: optuna.Trial) -> float:
        params = XGBParams(
            device=device,
            n_estimators=trial.suggest_int("n_estimators", 150, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        result = run_stratified_cv(X, y, params, n_splits=n_splits)
        return result.mean_metrics["roc_auc"]

    return objective


def train_final_model(X: pl.DataFrame, y: np.ndarray, params: XGBParams) -> xgb.XGBClassifier:
    """Fit on the full training set with the chosen (tuned) hyperparameters."""
    X_pd = X.to_pandas()
    spw = compute_scale_pos_weight(y)
    model = xgb.XGBClassifier(**params.to_xgb_kwargs(spw))
    model.fit(X_pd, y)
    return model
