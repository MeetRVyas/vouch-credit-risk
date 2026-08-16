"""FastAPI application factory.

The lifespan context is the composition root: it builds the expensive
singletons exactly once (model + SHAP explainer load, DB engine/session
factory) and hangs them off `app.state`, where `dependencies.py` retrieves
them per-request. Nothing below this point re-loads the model or re-opens
a DB connection per request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from credit_risk.api.config import Settings, get_settings
from credit_risk.api.db.session import init_models, make_engine, make_session_factory
from credit_risk.api.repositories.model_repository import FileModelRepository
from credit_risk.api.repositories.prediction_log_repository import (
    NullPredictionLogRepository,
    PostgresPredictionLogRepository,
)
from credit_risk.api.routers import health, predict
from credit_risk.api.services.prediction_service import PredictionService

logger = logging.getLogger("credit_risk.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings

        model_repository = FileModelRepository(settings.model_artifact_dir)
        logger.info("model artifact loaded (version=%s)", model_repository.get_artifact().model_version)

        if settings.log_predictions:
            engine = make_engine(settings)
            session_factory = make_session_factory(engine)
            try:
                await init_models(engine)
                log_repository = PostgresPredictionLogRepository(session_factory)
            except Exception:
                # Prediction logging is an audit trail, not the core value of
                # /predict -- a DB the API can't reach shouldn't take scoring
                # down with it. Degrade to a no-op logger and keep serving.
                logger.exception("could not initialize prediction-log DB; logging disabled for this run")
                log_repository = NullPredictionLogRepository()
                engine = None
        else:
            engine = None
            log_repository = NullPredictionLogRepository()

        app.state.prediction_service = PredictionService(
            model_repository=model_repository,
            log_repository=log_repository,
            risk_band_low_max=settings.risk_band_low_max,
            risk_band_medium_max=settings.risk_band_medium_max,
            log_predictions=settings.log_predictions,
        )

        yield

        if engine is not None:
            await engine.dispose()

    app = FastAPI(
        title="Credit Default Risk API",
        version="0.1.0",
        description="Probability-of-default scoring for loan applications.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predict.router)

    return app


app = create_app()
