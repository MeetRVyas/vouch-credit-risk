from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from credit_risk.api.db.models import PredictionLog
from credit_risk.api.repositories.base import PredictionLogEntry, PredictionLogRepository


class PostgresPredictionLogRepository(PredictionLogRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def save(self, entry: PredictionLogEntry) -> None:
        async with self._session_factory() as session:
            session.add(
                PredictionLog(
                    prediction_id=entry.prediction_id,
                    model_version=entry.model_version,
                    input_features=entry.input_features,
                    predicted_probability=entry.predicted_probability,
                    risk_band=entry.risk_band,
                )
            )
            await session.commit()


class NullPredictionLogRepository(PredictionLogRepository):
    """No-op logger. Used when LOG_PREDICTIONS=false or in tests that don't
    stand up a real Postgres -- keeps the service usable without a DB
    dependency, satisfying Liskov substitution against the same interface."""

    async def save(self, entry: PredictionLogEntry) -> None:
        return None
