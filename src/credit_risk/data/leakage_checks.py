"""Temporal-leakage checks for the bureau table.

This is deliberately a *separate, testable* module rather than inline
notebook code. The spec calls this out as the one risk to verify personally
rather than trust to AI-generated code -- so it needs to be something you can
read in one screen, run against real data, and reason about independently.

Two distinct things are checked:

1. Hard leakage (structural, provable): any bureau row with DAYS_CREDIT > 0
   is dated *after* the loan application it would be joined onto. Those rows
   must never enter the aggregation. This is enforced by the SQL's WHERE
   clause; this module re-checks it on the raw data and fails loudly if the
   assumption doesn't hold, rather than silently trusting the filter.

2. Soft leakage (structural, not fully fixable with two tables): bureau.csv
   is extracted at one fixed date for the whole table, not per-application.
   CREDIT_ACTIVE / AMT_CREDIT_SUM_DEBT reflect status *as of that extraction*,
   not as of each applicant's own application date. A credit that was open
   when someone applied may already read "Closed" if it closed before
   extraction. This can't be detected row-by-row from these two tables alone
   -- it's reported here as a quantified caveat (how much of the population
   has old applications relative to how "fresh" their bureau data looks),
   not swept under the rug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class LeakageReport:
    """Result of running the leakage checks against a raw bureau frame."""

    total_bureau_rows: int
    future_dated_rows: int
    future_dated_pct: float
    stale_update_rows: int
    stale_update_pct: float
    notes: list[str] = field(default_factory=list)

    @property
    def passed_hard_check(self) -> bool:
        """True if there are zero rows dated after the loan application."""
        return self.future_dated_rows == 0

    def summary(self) -> str:
        lines = [
            f"bureau rows scanned:              {self.total_bureau_rows:,}",
            f"rows with DAYS_CREDIT > 0:        {self.future_dated_rows:,} "
            f"({self.future_dated_pct:.3f}%)  -> "
            f"{'OK, none found' if self.passed_hard_check else 'MUST BE DROPPED BEFORE AGGREGATION'}",
            f"rows with stale DAYS_CREDIT_UPDATE (< -365d): {self.stale_update_rows:,} "
            f"({self.stale_update_pct:.2f}%)  -> informational, see notes",
        ]
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def check_bureau_temporal_integrity(bureau: pl.DataFrame) -> LeakageReport:
    """Run the hard + soft leakage checks against a raw (unfiltered) bureau frame.

    Parameters
    ----------
    bureau:
        The raw bureau table, unfiltered, with at least DAYS_CREDIT and
        DAYS_CREDIT_UPDATE columns.

    Raises
    ------
    AssertionError if required columns are missing.
    """
    required = {"SK_ID_CURR", "DAYS_CREDIT", "DAYS_CREDIT_UPDATE"}
    missing = required - set(bureau.columns)
    assert not missing, f"bureau frame is missing required columns: {missing}"

    total = bureau.height
    future_dated = bureau.filter(pl.col("DAYS_CREDIT") > 0).height
    stale = bureau.filter(pl.col("DAYS_CREDIT_UPDATE") < -365).height

    notes = [
        "AMT_CREDIT_SUM_DEBT / CREDIT_ACTIVE reflect bureau's single extraction "
        "date, not each applicant's own application date -- treat status-based "
        "features as an approximation, not a guarantee (see module docstring).",
    ]
    if stale:
        notes.append(
            f"{stale:,} rows have a bureau update older than 365 days before "
            "the reference date; their status fields are the least reliable "
            "and most likely to reflect stale, pre-extraction-gap information."
        )

    return LeakageReport(
        total_bureau_rows=total,
        future_dated_rows=future_dated,
        future_dated_pct=100 * future_dated / total if total else 0.0,
        stale_update_rows=stale,
        stale_update_pct=100 * stale / total if total else 0.0,
        notes=notes,
    )


def assert_no_future_dated_bureau_rows(bureau: pl.DataFrame) -> None:
    """Hard gate: raise if any bureau row is dated after its application.

    Call this right before feeding data into aggregation/training. It is
    intentionally strict (raises rather than warns) because this is the one
    bug class the spec calls out as too dangerous to leave to a linter.
    """
    report = check_bureau_temporal_integrity(bureau)
    if not report.passed_hard_check:
        raise ValueError(
            "Temporal leakage guard tripped: "
            f"{report.future_dated_rows} bureau row(s) have DAYS_CREDIT > 0 "
            "(dated after the loan application). Fix the source filter before "
            "training -- do not silently drop and continue.\n" + report.summary()
        )
