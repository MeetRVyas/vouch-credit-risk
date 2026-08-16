import polars as pl
import pytest

from credit_risk.data.duckdb_pipeline import load_and_join
from credit_risk.features.engineering import (
    add_ratio_features,
    build_feature_frame,
    clean,
    encode_categoricals,
    get_feature_columns,
)


@pytest.fixture(scope="module")
def joined_sample():
    return load_and_join(
        "data/sample/application_train_sample.csv",
        "data/sample/bureau_sample.csv",
    )


def test_clean_replaces_days_employed_sentinel_with_null():
    df = pl.DataFrame({"DAYS_EMPLOYED": [365243, -100, 365243, -50]})
    out = clean(df)
    assert out["DAYS_EMPLOYED"].null_count() == 2
    assert out["DAYS_EMPLOYED"].drop_nulls().to_list() == [-100, -50]


def test_add_ratio_features_produces_expected_columns_and_values():
    df = pl.DataFrame(
        {
            "AMT_CREDIT": [200_000.0],
            "AMT_INCOME_TOTAL": [100_000.0],
            "AMT_ANNUITY": [20_000.0],
            "DAYS_BIRTH": [-365 * 30],
            "DAYS_EMPLOYED": [-365 * 5],
            "bureau_total_credit_debt": [50_000.0],
            "bureau_total_credit_sum": [99_000.0],
            "bureau_active_credit_count": [2],
            "bureau_credit_count": [3],
        }
    )
    out = add_ratio_features(df)

    assert out["credit_to_income_ratio"][0] == pytest.approx(2.0)
    assert out["annuity_to_income_ratio"][0] == pytest.approx(0.2)
    assert out["annuity_to_credit_ratio"][0] == pytest.approx(0.1)
    assert out["age_years"][0] == pytest.approx(30.0, abs=0.05)
    assert out["bureau_debt_to_credit_ratio"][0] == pytest.approx(50_000 / (99_000 + 1.0))
    assert out["bureau_active_credit_share"][0] == pytest.approx(2 / 4)


def test_encode_categoricals_one_hot_expands_columns():
    df = pl.DataFrame({"CODE_GENDER": ["F", "M", "F"], "value": [1, 2, 3]})
    out = encode_categoricals(df)
    assert "CODE_GENDER_F" in out.columns
    assert "CODE_GENDER_M" in out.columns
    assert "CODE_GENDER" not in out.columns


def test_build_feature_frame_end_to_end_on_joined_sample(joined_sample):
    feat = build_feature_frame(joined_sample)
    assert feat.height == joined_sample.height
    cols = get_feature_columns(feat)
    assert "SK_ID_CURR" not in cols
    assert "TARGET" not in cols
    assert "credit_to_income_ratio" in cols
    # no categorical string columns should survive one-hot encoding
    dtypes = dict(zip(feat.columns, feat.dtypes, strict=True))
    for col in cols:
        assert dtypes[col] != pl.Utf8, f"{col} was not encoded to numeric"
