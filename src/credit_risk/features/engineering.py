"""Polars feature engineering: cleaning, encoding, and ratio features.

Runs after `duckdb_pipeline.load_and_join`, entirely in Polars, multi-threaded
over the full 300K+ row table. Produces a frame that's still Polars -- the
pandas conversion happens only at the XGBoost `.fit()` boundary
(see models/train.py), not here.
"""

from __future__ import annotations

import polars as pl

# The real dataset uses this literal sentinel for "currently employed: N/A"
# (pensioners / unemployed). Left as-is it's a massive outlier; treat it as
# missing instead.
_DAYS_EMPLOYED_SENTINEL = 365243

CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]

ID_AND_TARGET_COLUMNS = ["SK_ID_CURR", "TARGET"]


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Fix known data-quality quirks before feature engineering."""
    out = df.with_columns(
        pl.when(pl.col("DAYS_EMPLOYED") == _DAYS_EMPLOYED_SENTINEL)
        .then(None)
        .otherwise(pl.col("DAYS_EMPLOYED"))
        .alias("DAYS_EMPLOYED")
    )
    return out


def add_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer the ratio features called out in the spec, plus a couple of
    natural companions that reuse the same joined bureau aggregates."""
    out = df.with_columns(
        [
            (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias("credit_to_income_ratio"),
            (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("annuity_to_income_ratio"),
            (pl.col("AMT_ANNUITY") / pl.col("AMT_CREDIT")).alias("annuity_to_credit_ratio"),
            (-pl.col("DAYS_BIRTH") / 365.25).alias("age_years"),
            (-pl.col("DAYS_EMPLOYED") / 365.25).alias("years_employed"),
            (
                pl.col("bureau_total_credit_debt") / (pl.col("bureau_total_credit_sum") + 1.0)
            ).alias("bureau_debt_to_credit_ratio"),
            (
                pl.col("bureau_active_credit_count")
                / (pl.col("bureau_credit_count").cast(pl.Float64) + 1.0)
            ).alias("bureau_active_credit_share"),
        ]
    )
    return out


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot encode the low-cardinality categoricals; leave everything
    else numeric for XGBoost, which handles nulls natively."""
    return df.to_dummies(columns=[c for c in CATEGORICAL_COLUMNS if c in df.columns])


def build_feature_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Full feature-engineering pipeline: clean -> ratios -> encode."""
    return encode_categoricals(add_ratio_features(clean(df)))


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Every column except identifiers/target -- i.e. what the model trains on."""
    return [c for c in df.columns if c not in ID_AND_TARGET_COLUMNS]


def align_to_feature_columns(df: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Reindex an engineered frame to exactly the training-time feature columns,
    in order. Used at inference time, where a single request's one-hot
    encoding will only ever produce the categories present in that one row
    (e.g. `CODE_GENDER_F` but never `CODE_GENDER_M`) -- missing training
    columns are added as 0, and any columns the model never saw are dropped.
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(0).alias(c) for c in missing])
    return df.select(feature_columns)
