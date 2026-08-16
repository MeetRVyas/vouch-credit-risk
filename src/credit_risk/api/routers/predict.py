from __future__ import annotations

from fastapi import APIRouter

from credit_risk.api.dependencies import PredictionServiceDep
from credit_risk.api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, service: PredictionServiceDep) -> PredictionResponse:
    return await service.predict(
        applicant=request.applicant,
        bureau=request.bureau_summary,
        reference_id=request.reference_id,
    )
