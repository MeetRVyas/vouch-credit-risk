"""Model artifact packaging: bundles the fitted XGBoost model together with
the exact feature column order and a version tag, so the API never has to
guess what the model expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb


@dataclass
class ModelArtifact:
    model: xgb.XGBClassifier
    feature_columns: list[str]
    model_version: str

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out / "model.json"))
        (out / "metadata.json").write_text(
            json.dumps(
                {"feature_columns": self.feature_columns, "model_version": self.model_version},
                indent=2,
            )
        )

    @classmethod
    def load(cls, in_dir: str) -> ModelArtifact:
        in_path = Path(in_dir)
        model = xgb.XGBClassifier()
        model.load_model(str(in_path / "model.json"))
        metadata = json.loads((in_path / "metadata.json").read_text())
        return cls(
            model=model,
            feature_columns=metadata["feature_columns"],
            model_version=metadata["model_version"],
        )
