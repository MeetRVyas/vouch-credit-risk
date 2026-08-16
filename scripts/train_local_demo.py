"""Train a small demo model on the synthetic sample data and save the artifact
the API loads. This is NOT the real training run (that's Optuna + MLflow +
Kaggle GPU + the real dataset, see notebooks/01_pd_model_pipeline.ipynb) --
it exists so `credit_risk.api` has a genuine model.json to serve locally and
in CI smoke tests, without depending on Kaggle or a multi-hour tuning search.

Usage:
    python scripts/train_local_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from credit_risk.data.duckdb_pipeline import load_and_join  # noqa: E402
from credit_risk.data.synthetic import write_sample_csvs  # noqa: E402
from credit_risk.evaluation.metrics import (  # noqa: E402
    approval_cutoff_table,
    compute_classification_metrics,
)
from credit_risk.features.engineering import build_feature_frame, get_feature_columns  # noqa: E402
from credit_risk.models.artifact import ModelArtifact  # noqa: E402
from credit_risk.models.train import XGBParams, train_final_model  # noqa: E402

SAMPLE_DIR = REPO_ROOT / "data" / "sample"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "model"
MODEL_VERSION = "demo-0.1.0"


def main() -> None:
    app_path = SAMPLE_DIR / "application_train_sample.csv"
    bureau_path = SAMPLE_DIR / "bureau_sample.csv"
    if not app_path.exists() or not bureau_path.exists():
        print("sample data not found, generating it now...")
        write_sample_csvs(str(SAMPLE_DIR))

    joined = load_and_join(str(app_path), str(bureau_path))
    feat = build_feature_frame(joined)
    feature_cols = get_feature_columns(feat)
    X = feat.select(feature_cols)
    y = feat["TARGET"].to_numpy()

    params = XGBParams(n_estimators=200, max_depth=4, learning_rate=0.05)
    model = train_final_model(X, y, params)

    artifact = ModelArtifact(model=model, feature_columns=feature_cols, model_version=MODEL_VERSION)
    artifact.save(str(ARTIFACT_DIR))

    # in-sample metrics, for demo purposes only -- NOT a substitute for the
    # held-out CV metrics the real training run reports (see README).
    probs = model.predict_proba(X.to_pandas())[:, 1]
    metrics = compute_classification_metrics(y, probs).as_dict()
    cutoffs = approval_cutoff_table(y, probs)

    (ARTIFACT_DIR / "demo_metrics.json").write_text(
        json.dumps({"in_sample_metrics": metrics, "approval_cutoffs": cutoffs}, indent=2)
    )

    print(f"saved model artifact to {ARTIFACT_DIR} (version={MODEL_VERSION})")
    print("in-sample metrics (demo data, NOT held-out):", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
