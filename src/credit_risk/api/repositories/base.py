"""Repository interfaces. The service layer depends on these abstractions,
not on concrete Postgres/filesystem implementations (dependency inversion) --
this is what makes the prediction service testable without a real DB and
swappable if the logging store or model store ever changes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from credit_risk.models.artifact import ModelArtifact


@dataclass
class PredictionLogEntry:
    prediction_id: str
    model_version: str
    input_features: dict
    predicted_probability: float
    risk_band: str


class ModelRepository(ABC):
    @abstractmethod
    def get_artifact(self) -> ModelArtifact: ...


class PredictionLogRepository(ABC):
    @abstractmethod
    async def save(self, entry: PredictionLogEntry) -> None: ...
