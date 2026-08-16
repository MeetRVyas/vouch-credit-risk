"""FastAPI dependency providers.

Expensive singletons (the loaded model, the DB session factory, the
prediction service) are built once in `main.py`'s lifespan and stashed on
`app.state`; these `get_*` functions just retrieve them per-request. This
keeps route handlers thin and keeps the wiring in exactly one place.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from credit_risk.api.services.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
