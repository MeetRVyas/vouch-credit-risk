"""Load application_train + bureau via DuckDB and aggregate bureau in SQL.

Why DuckDB here and Polars downstream (see README / spec sec. 3): bureau is
one-to-many per applicant, so collapsing it is a GROUP BY/JOIN by nature --
that's what SQL is for. Everything after the join (encoding, ratio features)
is per-row transform work, which is what Polars is for. This module owns the
boundary: it hands back a single Polars DataFrame, one row per SK_ID_CURR,
with application columns + `bureau_*` aggregate columns attached.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import duckdb
import polars as pl

from credit_risk.data.leakage_checks import assert_no_future_dated_bureau_rows
from credit_risk.features.engineering import ID_AND_TARGET_COLUMNS, RAW_APPLICATION_FEATURE_COLUMNS

_SQL_PATH = Path(__file__).resolve().parents[3] / "sql" / "bureau_aggregation.sql"


def _load_aggregation_sql() -> str:
    if _SQL_PATH.exists():
        return _SQL_PATH.read_text()
    # fallback for packaged installs where the repo-root sql/ dir isn't shipped
    return resources.files("credit_risk").joinpath("../../sql/bureau_aggregation.sql").read_text()


def load_and_join(application_path: str, bureau_path: str, *, verify_leakage: bool = True) -> pl.DataFrame:
    """Read the two CSVs, aggregate bureau in DuckDB, and left-join onto application.

    Parameters
    ----------
    application_path, bureau_path:
        Paths to application_train.csv and bureau.csv (or the sample stand-ins).
        Works unmodified against the real Kaggle files -- e.g. on Kaggle:
        `/kaggle/input/home-credit-default-risk/application_train.csv`.
    verify_leakage:
        Run the hard temporal-leakage check on the raw bureau data before
        aggregating. Leave this on unless you've already verified the source
        file elsewhere -- it's cheap and it's the one check the spec singles
        out as worth never skipping.
    """
    con = duckdb.connect(database=":memory:")
    con.execute(f"create view application_train as select * from read_csv_auto('{application_path}')")
    con.execute(f"create view bureau as select * from read_csv_auto('{bureau_path}')")

    if verify_leakage:
        raw_bureau = con.execute("select * from bureau").pl()
        assert_no_future_dated_bureau_rows(raw_bureau)

    bureau_agg = con.execute(_load_aggregation_sql()).pl()

    application_select_cols = ", ".join(ID_AND_TARGET_COLUMNS + RAW_APPLICATION_FEATURE_COLUMNS)
    application = con.execute(f"select {application_select_cols} from application_train").pl()
    con.close()

    joined = application.join(bureau_agg, on="SK_ID_CURR", how="left")

    # defensive: DuckDB's SUM() can widen integer aggregates to
    # HUGEINT/DECIMAL, which becomes an unusable "object" dtype after a
    # pandas conversion downstream. Normalize any Decimal columns to Float64
    # here rather than relying solely on the explicit casts in the SQL file.
    decimal_cols = [
        name for name, dtype in zip(joined.columns, joined.dtypes, strict=True) if dtype.base_type() == pl.Decimal
    ]
    if decimal_cols:
        joined = joined.with_columns([pl.col(c).cast(pl.Float64) for c in decimal_cols])

    # applicants with zero prior bureau records get nulls from the left join --
    # that's a meaningful state ("no bureau history"), not missing data, so we
    # fill counts/sums with 0 and leave rate-like columns (avg overdue, etc.) as
    # null for the imputer to handle explicitly downstream.
    count_and_sum_cols = [
        c
        for c in bureau_agg.columns
        if c.startswith("bureau_") and ("count" in c or "sum" in c or "total" in c or "prolong" in c)
    ]
    joined = joined.with_columns([pl.col(c).fill_null(0) for c in count_and_sum_cols])

    return joined
