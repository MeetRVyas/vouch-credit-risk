"""Pydantic request/response models.

Two design choices worth calling out:

1. `ApplicantProfile` exposes natural units (age_years, years_employed)
   rather than the raw dataset's DAYS_BIRTH/DAYS_EMPLOYED negative-day-count
   convention -- that's a Kaggle-CSV implementation detail, not something an
   API caller should have to know. The service layer converts back to the
   training representation before scoring.

2. `BureauSummary` is optional and defaults to "no prior credit history"
   (all zeros). That default is a real, meaningful state -- a thin-file
   applicant -- not a placeholder, which matters for an underbanked-lending
   use case.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ApplicantProfile(BaseModel):
    name_contract_type: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    code_gender: Literal["M", "F"] = "F"
    flag_own_car: Literal["Y", "N"] = "N"
    flag_own_realty: Literal["Y", "N"] = "N"
    cnt_children: int = Field(0, ge=0, le=20)
    cnt_fam_members: float = Field(1.0, ge=1, le=20)

    amt_income_total: float = Field(..., gt=0, description="Annual income")
    amt_credit: float = Field(..., gt=0, description="Requested loan amount")
    amt_annuity: float = Field(..., gt=0, description="Loan annuity (periodic payment)")
    amt_goods_price: float | None = Field(None, gt=0, description="Price of goods being financed, if applicable")

    name_income_type: str = "Working"
    name_education_type: str = "Secondary / secondary special"
    name_family_status: str = "Married"
    name_housing_type: str = "House / apartment"

    age_years: float = Field(..., ge=18, le=100)
    years_employed: float | None = Field(
        None, ge=0, le=80, description="Omit if unemployed / retired / not applicable"
    )

    region_rating_client: Literal[1, 2, 3] = 2

    ext_source_1: float | None = Field(None, ge=0, le=1, description="External scoring source 1, if available")
    ext_source_2: float | None = Field(None, ge=0, le=1, description="External scoring source 2, if available")
    ext_source_3: float | None = Field(None, ge=0, le=1, description="External scoring source 3, if available")


class BureauSummary(BaseModel):
    """Pre-aggregated prior-credit-bureau history. Defaults represent an
    applicant with no bureau record on file."""

    bureau_credit_count: int = 0
    bureau_distinct_credit_types: int = 0
    bureau_active_credit_count: int = 0
    bureau_closed_credit_count: int = 0
    bureau_bad_debt_count: int = 0
    bureau_avg_days_overdue: float = 0.0
    bureau_max_days_overdue: float = 0.0
    bureau_credits_ever_overdue: int = 0
    bureau_total_prolongations: int = 0
    bureau_total_credit_sum: float = 0.0
    bureau_total_credit_debt: float = 0.0
    bureau_total_credit_limit: float = 0.0
    bureau_total_overdue_amount: float = 0.0
    bureau_max_overdue_amount_ever: float = 0.0
    bureau_days_since_oldest_credit: float | None = None
    bureau_days_since_most_recent_credit: float | None = None
    bureau_avg_days_since_credit: float | None = None
    bureau_days_since_last_bureau_update: float | None = None


class PredictionRequest(BaseModel):
    applicant: ApplicantProfile
    bureau_summary: BureauSummary = Field(default_factory=BureauSummary)
    reference_id: str | None = Field(
        None, description="Caller's own correlation id. Echoed back, never persisted in the prediction log."
    )


class RiskFactor(BaseModel):
    feature: str
    direction: Literal["increases_risk", "decreases_risk"]
    contribution: float


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_probability: float
    risk_band: Literal["low", "medium", "high"]
    model_version: str
    top_risk_factors: list[RiskFactor]
    reference_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
