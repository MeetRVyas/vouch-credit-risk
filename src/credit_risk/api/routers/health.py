from __future__ import annotations

from fastapi import APIRouter, Request

from credit_risk.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        return HealthResponse(status="degraded", model_loaded=False)

    artifact = service._model_repository.get_artifact()  # noqa: SLF001 -- health check reads internal state deliberately
    return HealthResponse(status="ok", model_loaded=True, model_version=artifact.model_version)
