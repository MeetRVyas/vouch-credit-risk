"""Centralized, env-driven settings. Single source of truth for config so no
other module reaches into os.environ directly (SOLID: single responsibility)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    app_env: str = "development"

    # Model artifact produced by scripts/train_local_demo.py or the real
    # training notebook -- a directory containing model.json + metadata.json.
    model_artifact_dir: str = "artifacts/model"

    # Prediction logging (spec sec. 5c): a *separate* Postgres from MLflow's
    # tracking DB, different lifecycle/consumer. Async URL, e.g.:
    #   postgresql+asyncpg://user:pass@localhost:5432/credit_risk
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/credit_risk"
    log_predictions: bool = True

    # Risk bands are a placeholder split until calibrated against the real
    # training run's approval_cutoff_table (see evaluation/metrics.py) --
    # override via env once real thresholds are known.
    risk_band_low_max: float = 0.10
    risk_band_medium_max: float = 0.30

    cors_allow_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


def get_settings() -> Settings:
    return Settings()
