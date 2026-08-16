"""Prediction service: the one place that knows how to turn a request into
a scored, explained, (optionally) logged prediction.

Depends only on the repository *interfaces* (ModelRepository,
PredictionLogRepository), not concrete Postgres/filesystem classes --
dependency inversion, so this class is unit-testable with fakes and doesn't
care whether logging is Postgres, a no-op, or something else later.
"""

from __future__ import annotations

import uuid

import polars as pl

from credit_risk.api.repositories.base import (
    ModelRepository,
    PredictionLogEntry,
    PredictionLogRepository,
)
from credit_risk.api.schemas import ApplicantProfile, BureauSummary, PredictionResponse, RiskFactor
from credit_risk.explain.shap_explain import (
    build_explainer,
    compute_shap_values,
    top_contributors_for_row,
)
from credit_risk.features.engineering import align_to_feature_columns, build_feature_frame

_DAYS_PER_YEAR = 365.25
_DAYS_EMPLOYED_SENTINEL = 365243  # matches the real dataset's "not applicable" sentinel
_DEFAULT_EXT_SOURCE = 0.5  # neutral prior when a bureau/external score isn't supplied

# Bureau "recency" fields are the only ones a caller might genuinely omit
# (None) *and* that can end up as the sole value in a single-row frame --
# without an explicit dtype, Polars infers those all-None columns as an
# untyped Null column, which XGBoost then rejects. Cast them explicitly so a
# missing value still comes through as a proper (numeric) null.
_NULLABLE_FLOAT_COLUMNS = [
    "bureau_days_since_oldest_credit",
    "bureau_days_since_most_recent_credit",
    "bureau_avg_days_since_credit",
    "bureau_days_since_last_bureau_update",
]


def _request_to_raw_frame(applicant: ApplicantProfile, bureau: BureauSummary) -> pl.DataFrame:
    """Map the API's natural-units request schema onto the raw,
    training-time column names/units, so it can flow through the exact same
    `credit_risk.features.engineering` pipeline used at training time."""
    days_birth = -round(applicant.age_years * _DAYS_PER_YEAR)
    days_employed = (
        _DAYS_EMPLOYED_SENTINEL
        if applicant.years_employed is None
        else -round(applicant.years_employed * _DAYS_PER_YEAR)
    )
    goods_price = applicant.amt_goods_price if applicant.amt_goods_price is not None else applicant.amt_credit

    row = {
        "NAME_CONTRACT_TYPE": applicant.name_contract_type,
        "CODE_GENDER": applicant.code_gender,
        "FLAG_OWN_CAR": applicant.flag_own_car,
        "FLAG_OWN_REALTY": applicant.flag_own_realty,
        "CNT_CHILDREN": applicant.cnt_children,
        "CNT_FAM_MEMBERS": applicant.cnt_fam_members,
        "AMT_INCOME_TOTAL": applicant.amt_income_total,
        "AMT_CREDIT": applicant.amt_credit,
        "AMT_ANNUITY": applicant.amt_annuity,
        "AMT_GOODS_PRICE": goods_price,
        "NAME_INCOME_TYPE": applicant.name_income_type,
        "NAME_EDUCATION_TYPE": applicant.name_education_type,
        "NAME_FAMILY_STATUS": applicant.name_family_status,
        "NAME_HOUSING_TYPE": applicant.name_housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "REGION_RATING_CLIENT": applicant.region_rating_client,
        "EXT_SOURCE_1": applicant.ext_source_1 if applicant.ext_source_1 is not None else _DEFAULT_EXT_SOURCE,
        "EXT_SOURCE_2": applicant.ext_source_2 if applicant.ext_source_2 is not None else _DEFAULT_EXT_SOURCE,
        "EXT_SOURCE_3": applicant.ext_source_3 if applicant.ext_source_3 is not None else _DEFAULT_EXT_SOURCE,
        **bureau.model_dump(),
    }
    df = pl.DataFrame([row])
    return df.with_columns([pl.col(c).cast(pl.Float64) for c in _NULLABLE_FLOAT_COLUMNS])


class PredictionService:
    def __init__(
        self,
        model_repository: ModelRepository,
        log_repository: PredictionLogRepository,
        risk_band_low_max: float,
        risk_band_medium_max: float,
        log_predictions: bool = True,
    ):
        self._model_repository = model_repository
        self._log_repository = log_repository
        self._risk_band_low_max = risk_band_low_max
        self._risk_band_medium_max = risk_band_medium_max
        self._log_predictions = log_predictions

        artifact = self._model_repository.get_artifact()
        self._explainer = build_explainer(artifact.model)

    def _risk_band(self, probability: float) -> str:
        if probability < self._risk_band_low_max:
            return "low"
        if probability < self._risk_band_medium_max:
            return "medium"
        return "high"

    async def predict(
        self, applicant: ApplicantProfile, bureau: BureauSummary, reference_id: str | None
    ) -> PredictionResponse:
        artifact = self._model_repository.get_artifact()

        raw = _request_to_raw_frame(applicant, bureau)
        engineered = build_feature_frame(raw)
        aligned = align_to_feature_columns(engineered, artifact.feature_columns)
        X_pd = aligned.to_pandas()

        probability = float(artifact.model.predict_proba(X_pd)[:, 1][0])
        risk_band = self._risk_band(probability)

        shap_values = compute_shap_values(self._explainer, X_pd)
        contributors = top_contributors_for_row(shap_values, X_pd, row_idx=0, top_n=5)
        top_factors = [
            RiskFactor(feature=c["feature"], direction=c["direction"], contribution=round(c["shap_contribution"], 4))
            for c in contributors
        ]

        prediction_id = str(uuid.uuid4())

        if self._log_predictions:
            await self._log_repository.save(
                PredictionLogEntry(
                    prediction_id=prediction_id,
                    model_version=artifact.model_version,
                    input_features={**applicant.model_dump(), **bureau.model_dump()},
                    predicted_probability=probability,
                    risk_band=risk_band,
                )
            )

        return PredictionResponse(
            prediction_id=prediction_id,
            predicted_probability=round(probability, 6),
            risk_band=risk_band,
            model_version=artifact.model_version,
            top_risk_factors=top_factors,
            reference_id=reference_id,
        )
