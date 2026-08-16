from __future__ import annotations

from credit_risk.api.repositories.base import ModelRepository
from credit_risk.models.artifact import ModelArtifact


class FileModelRepository(ModelRepository):
    """Loads the model artifact from disk once (at construction, which
    happens a single time in the composition root -- see dependencies.py)
    and caches it in memory. XGBoost model + SHAP explainer construction
    aren't cheap, and the artifact is immutable for the process lifetime
    (a new deploy = a new process, given the "no registry, no staging"
    scoping decision)."""

    def __init__(self, artifact_dir: str):
        self._artifact = ModelArtifact.load(artifact_dir)

    def get_artifact(self) -> ModelArtifact:
        return self._artifact
