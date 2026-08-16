import polars as pl
import pytest

from credit_risk.data.leakage_checks import (
    assert_no_future_dated_bureau_rows,
    check_bureau_temporal_integrity,
)


def _bureau_frame(days_credit_values, days_update_values=None):
    n = len(days_credit_values)
    if days_update_values is None:
        days_update_values = [-10] * n
    return pl.DataFrame(
        {
            "SK_ID_CURR": list(range(n)),
            "DAYS_CREDIT": days_credit_values,
            "DAYS_CREDIT_UPDATE": days_update_values,
        }
    )


def test_clean_bureau_passes_hard_check():
    df = _bureau_frame([-10, -200, -3000])
    report = check_bureau_temporal_integrity(df)
    assert report.passed_hard_check
    assert report.future_dated_rows == 0


def test_future_dated_rows_fail_hard_check():
    df = _bureau_frame([-10, 5, -3000, 42])
    report = check_bureau_temporal_integrity(df)
    assert not report.passed_hard_check
    assert report.future_dated_rows == 2
    assert report.future_dated_pct == pytest.approx(50.0)


def test_assert_raises_on_future_dated_rows():
    df = _bureau_frame([-10, 5])
    with pytest.raises(ValueError, match="Temporal leakage guard tripped"):
        assert_no_future_dated_bureau_rows(df)


def test_assert_does_not_raise_on_clean_data():
    df = _bureau_frame([-10, -20, -30])
    assert_no_future_dated_bureau_rows(df)  # should not raise


def test_stale_update_rows_are_counted_but_not_fatal():
    df = _bureau_frame([-10, -10], days_update_values=[-400, -10])
    report = check_bureau_temporal_integrity(df)
    assert report.passed_hard_check
    assert report.stale_update_rows == 1


def test_missing_columns_raises_assertion_error():
    df = pl.DataFrame({"SK_ID_CURR": [1, 2]})
    with pytest.raises(AssertionError):
        check_bureau_temporal_integrity(df)
