-- bureau_aggregation.sql
--
-- Collapses `bureau` (one-to-many: multiple prior-credit rows per SK_ID_CURR)
-- into one row per SK_ID_CURR, ready to left-join onto application_train.
--
-- Executed against two views the Python loader registers before running this
-- file: `bureau` and `application_train`. See src/credit_risk/data/duckdb_pipeline.py.
--
-- --- Temporal-leakage note (see also src/credit_risk/data/leakage_checks.py) ---
-- DAYS_CREDIT is defined as "days before the loan application that this bureau
-- credit was applied for" and is expected to be <= 0 for every row. We filter
-- to DAYS_CREDIT <= 0 defensively -- any positive value would mean a bureau
-- record dated *after* the application, which the model must never see.
-- This does NOT fully solve the dataset's known subtlety: CREDIT_ACTIVE and
-- AMT_CREDIT_SUM_DEBT reflect bureau's status as of the table's single
-- extraction date, not each applicant's own application date, so a credit
-- that was still open when someone applied may show "Closed" here if it
-- closed before extraction. That's a real, documented limitation of this
-- two-table scope -- flagged in the README, not hidden.

with prior_credit as (
    select
        SK_ID_CURR,
        SK_ID_BUREAU,
        CREDIT_ACTIVE,
        CREDIT_TYPE,
        DAYS_CREDIT,
        DAYS_CREDIT_ENDDATE,
        DAYS_ENDDATE_FACT,
        DAYS_CREDIT_UPDATE,
        CREDIT_DAY_OVERDUE,
        CNT_CREDIT_PROLONG,
        coalesce(AMT_CREDIT_SUM, 0)        as AMT_CREDIT_SUM,
        coalesce(AMT_CREDIT_SUM_DEBT, 0)   as AMT_CREDIT_SUM_DEBT,
        coalesce(AMT_CREDIT_SUM_LIMIT, 0)  as AMT_CREDIT_SUM_LIMIT,
        coalesce(AMT_CREDIT_SUM_OVERDUE, 0) as AMT_CREDIT_SUM_OVERDUE,
        coalesce(AMT_CREDIT_MAX_OVERDUE, 0) as AMT_CREDIT_MAX_OVERDUE
    from bureau
    where DAYS_CREDIT <= 0   -- leakage guard: drop any bureau row dated after the application
)
select
    SK_ID_CURR,

    -- volume / breadth of prior credit history
    count(*)                                            as bureau_credit_count,
    count(distinct CREDIT_TYPE)                         as bureau_distinct_credit_types,
    count(*) filter (where CREDIT_ACTIVE = 'Active')     as bureau_active_credit_count,
    count(*) filter (where CREDIT_ACTIVE = 'Closed')     as bureau_closed_credit_count,
    count(*) filter (where CREDIT_ACTIVE = 'Bad debt')   as bureau_bad_debt_count,

    -- delinquency signal
    avg(CREDIT_DAY_OVERDUE)                             as bureau_avg_days_overdue,
    max(CREDIT_DAY_OVERDUE)                             as bureau_max_days_overdue,
    -- DuckDB's default SUM(integer) widens to HUGEINT/DECIMAL for overflow
    -- safety; cast back to BIGINT so downstream pandas/XGBoost see a plain
    -- numeric dtype instead of "object".
    sum(case when CREDIT_DAY_OVERDUE > 0 then 1 else 0 end)::BIGINT as bureau_credits_ever_overdue,
    sum(CNT_CREDIT_PROLONG)::BIGINT                     as bureau_total_prolongations,

    -- exposure / balances
    sum(AMT_CREDIT_SUM)                                 as bureau_total_credit_sum,
    sum(AMT_CREDIT_SUM_DEBT)                            as bureau_total_credit_debt,
    sum(AMT_CREDIT_SUM_LIMIT)                           as bureau_total_credit_limit,
    sum(AMT_CREDIT_SUM_OVERDUE)                         as bureau_total_overdue_amount,
    max(AMT_CREDIT_MAX_OVERDUE)                         as bureau_max_overdue_amount_ever,

    -- recency / tenure, all still expressed in the source's negative-days convention
    min(DAYS_CREDIT)                                    as bureau_days_since_oldest_credit,
    max(DAYS_CREDIT)                                    as bureau_days_since_most_recent_credit,
    avg(DAYS_CREDIT)                                     as bureau_avg_days_since_credit,
    max(DAYS_CREDIT_UPDATE)                             as bureau_days_since_last_bureau_update

from prior_credit
group by SK_ID_CURR
