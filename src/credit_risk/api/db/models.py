"""SQLAlchemy models for the prediction-logging Postgres DB.

This is a separate database from MLflow's tracking store (spec sec. 3/5c):
different lifecycle, different consumer (this one is an audit/monitoring
trail for served predictions, not an experiment record).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """One row per /predict call. No identity fields, per spec sec. 5c --
    only what's needed to monitor model behavior and feed the PSI check."""

    __tablename__ = "prediction_logs"

    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    model_version: Mapped[str] = mapped_column(String(64))
    input_features: Mapped[dict] = mapped_column(JSON)
    predicted_probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16))
