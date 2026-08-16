"""Schema-matched synthetic data, standing in for the real Kaggle CSVs.

The real `application_train.csv` / `bureau.csv` come from a competition
dataset that this sandbox has no network path to (Kaggle isn't reachable
here) and that shouldn't be committed to the repo either way. Every other
module in this package is written against the *real* column names and
semantics of those two files, so it runs unmodified against the genuine
CSVs on Kaggle -- this generator exists purely so the pipeline, API, and
tests have something schema-correct to run against locally.

Column set is a deliberate subset of the ~122 application_train columns:
enough to exercise ratio features, categoricals, and EXT_SOURCE-style
scores without carrying dead weight.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_DEFAULT_RATE = 0.08  # matches the spec's ~8% figure for the real dataset


def make_synthetic_application_train(n_applicants: int = 2000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    income = rng.lognormal(mean=11.9, sigma=0.5, size=n_applicants).round(-2)
    credit = (income * rng.uniform(1.5, 6.0, size=n_applicants)).round(-2)
    annuity = (credit / rng.uniform(8, 30, size=n_applicants)).round(2)
    goods_price = (credit * rng.uniform(0.85, 1.0, size=n_applicants)).round(-2)

    days_birth = -rng.integers(20 * 365, 69 * 365, size=n_applicants)
    days_employed = np.where(
        rng.random(n_applicants) < 0.85,
        -rng.integers(0, 40 * 365, size=n_applicants),
        365243,  # real dataset's literal sentinel for "not employed / pensioner"
    )

    ext_source_1 = np.clip(rng.normal(0.5, 0.18, size=n_applicants), 0, 1)
    ext_source_2 = np.clip(rng.normal(0.5, 0.18, size=n_applicants), 0, 1)
    ext_source_3 = np.clip(rng.normal(0.5, 0.18, size=n_applicants), 0, 1)

    # target correlated with the external scores + leverage, plus noise, so the
    # pipeline has genuine (if modest) signal to fit -- not pure random noise.
    leverage = credit / income
    risk_score = (
        -3.0
        - 2.5 * (ext_source_1 - 0.5)
        - 2.0 * (ext_source_2 - 0.5)
        - 1.5 * (ext_source_3 - 0.5)
        + 0.35 * (leverage - leverage.mean()) / leverage.std()
        + rng.normal(0, 1, size=n_applicants)
    )
    p_default = 1 / (1 + np.exp(-risk_score))
    # rescale so the base rate matches the real dataset's ~8%
    p_default = p_default * (_DEFAULT_RATE / p_default.mean())
    target = rng.binomial(1, np.clip(p_default, 0, 0.95))

    df = pl.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_001, 100_001 + n_applicants),
            "TARGET": target.astype(np.int8),
            "NAME_CONTRACT_TYPE": rng.choice(
                ["Cash loans", "Revolving loans"], size=n_applicants, p=[0.9, 0.1]
            ),
            "CODE_GENDER": rng.choice(["F", "M"], size=n_applicants, p=[0.65, 0.35]),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=n_applicants, p=[0.35, 0.65]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=n_applicants, p=[0.7, 0.3]),
            "CNT_CHILDREN": rng.poisson(0.4, size=n_applicants).astype(np.int64),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods_price,
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
                size=n_applicants,
                p=[0.51, 0.23, 0.18, 0.07, 0.01],
            ),
            "NAME_EDUCATION_TYPE": rng.choice(
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                ],
                size=n_applicants,
                p=[0.71, 0.24, 0.03, 0.02],
            ),
            "NAME_FAMILY_STATUS": rng.choice(
                ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"],
                size=n_applicants,
                p=[0.64, 0.15, 0.1, 0.06, 0.05],
            ),
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "With parents", "Rented apartment", "Municipal apartment"],
                size=n_applicants,
                p=[0.88, 0.05, 0.04, 0.03],
            ),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "CNT_FAM_MEMBERS": (1 + rng.poisson(1.0, size=n_applicants)).astype(np.float64),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], size=n_applicants, p=[0.15, 0.65, 0.2]),
            "EXT_SOURCE_1": ext_source_1,
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3,
        }
    )
    return df


def make_synthetic_bureau(application: pl.DataFrame, seed: int = 43) -> pl.DataFrame:
    """Generate a one-to-many bureau table keyed on the applicants above.

    Row count per applicant, credit types, and overdue behavior are loosely
    correlated with TARGET so aggregated bureau features carry real signal.
    """
    rng = np.random.default_rng(seed)
    ids = application["SK_ID_CURR"].to_numpy()
    targets = application["TARGET"].to_numpy()

    credit_types = [
        "Consumer credit",
        "Credit card",
        "Car loan",
        "Mortgage",
        "Microloan",
    ]

    rows: list[dict] = []
    sk_id_bureau = 5_000_001
    for sk_id_curr, target in zip(ids, targets, strict=True):
        n_credits = rng.poisson(2.2 + 3.0 * target)  # riskier applicants -> more prior credit lines
        for _ in range(n_credits):
            days_credit = -int(rng.integers(30, 3000))
            active = rng.random() < (0.35 + 0.25 * target)
            overdue_days = int(rng.integers(0, 120)) if rng.random() < (0.05 + 0.25 * target) else 0
            credit_sum = float(rng.lognormal(mean=10.5, sigma=0.7))
            debt_ratio = rng.uniform(0, 0.9) if active else 0.0

            rows.append(
                {
                    "SK_ID_CURR": int(sk_id_curr),
                    "SK_ID_BUREAU": sk_id_bureau,
                    "CREDIT_ACTIVE": "Active" if active else rng.choice(["Closed", "Bad debt"], p=[0.97, 0.03]),
                    "CREDIT_CURRENCY": "currency 1",
                    "DAYS_CREDIT": days_credit,
                    "CREDIT_DAY_OVERDUE": overdue_days,
                    "DAYS_CREDIT_ENDDATE": days_credit + int(rng.integers(180, 1800)),
                    "DAYS_ENDDATE_FACT": None if active else days_credit + int(rng.integers(90, 1500)),
                    "AMT_CREDIT_MAX_OVERDUE": float(overdue_days * rng.uniform(50, 500)),
                    "CNT_CREDIT_PROLONG": int(rng.poisson(0.05)),
                    "AMT_CREDIT_SUM": round(credit_sum, 2),
                    "AMT_CREDIT_SUM_DEBT": round(credit_sum * debt_ratio, 2),
                    "AMT_CREDIT_SUM_LIMIT": round(credit_sum * rng.uniform(0, 0.2), 2),
                    "AMT_CREDIT_SUM_OVERDUE": round(overdue_days * rng.uniform(0, 30), 2),
                    "CREDIT_TYPE": rng.choice(credit_types, p=[0.55, 0.25, 0.08, 0.07, 0.05]),
                    "DAYS_CREDIT_UPDATE": days_credit + int(rng.integers(0, min(200, -days_credit + 1))),
                    "AMT_ANNUITY": round(float(rng.lognormal(mean=7.5, sigma=0.8)), 2)
                    if rng.random() < 0.6
                    else None,
                }
            )
            sk_id_bureau += 1

    return pl.DataFrame(rows)


def write_sample_csvs(out_dir: str, n_applicants: int = 2000, seed: int = 42) -> tuple[str, str]:
    """Write application_train_sample.csv + bureau_sample.csv to out_dir. Returns paths."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    app = make_synthetic_application_train(n_applicants=n_applicants, seed=seed)
    bureau = make_synthetic_bureau(app, seed=seed + 1)

    app_path = os.path.join(out_dir, "application_train_sample.csv")
    bureau_path = os.path.join(out_dir, "bureau_sample.csv")
    app.write_csv(app_path)
    bureau.write_csv(bureau_path)
    return app_path, bureau_path


if __name__ == "__main__":
    paths = write_sample_csvs("data/sample", n_applicants=2000)
    print("wrote:", paths)
