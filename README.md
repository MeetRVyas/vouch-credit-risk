# Credit Default Risk — Probability-of-Default Model

An end-to-end probability-of-default (PD) model on real lending data:
DuckDB + Polars feature pipeline, stratified-CV XGBoost tuned with Optuna,
tracked in a self-hosted MLflow server, explained with SHAP, and served
behind a SOLID-structured async FastAPI backend with a basic React frontend.

1-day AI-assisted ("vibe-coded") build with manual review, not a multi-week research project.

## Dataset

**[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** (Kaggle, free competition dataset). Home Credit's business — lending to underbanked populations using alternative data instead of a traditional credit history.

| Table | Role |
|---|---|
| `application_train.csv` | Main table — one row per loan, binary `TARGET`, keyed on `SK_ID_CURR` |
| `bureau.csv` | Applicant's prior credit history from other institutions — multiple rows per `SK_ID_CURR`, aggregated before joining |

## Architecture

| Layer | Tool | |
|---|---|---|
| Bureau aggregation | **DuckDB (SQL)** | `bureau.csv` is one-to-many — collapsing it to one row per applicant is a `GROUP BY`/`JOIN` by nature |
| Feature engineering | **Polars** | Multi-threaded, memory-efficient at 300K+ rows |
| Model-fit boundary | **pandas** | XGBoost's sklearn API is most stable on pandas/numpy at `.fit()` |
| Training | **XGBoost**, `tree_method="hist"` | |
| Tuning | **Optuna** | Sample-efficient Bayesian search under a limited budget |
| Experiment tracking | **MLflow**, self-hosted | Reproducible record across trials, survives past Kaggle's ephemeral session |
| Interpretability | **SHAP** | Adverse-action-style explainability |
| Serving | **FastAPI**, async | `/predict`, `/health` |
| Prediction logging | **Postgres**, separate from MLflow's tracking DB | |
| Frontend | **React + TypeScript (Vite)** | Basic client for `/predict` |
| Deployment | **Docker → HF Spaces** | |
| CI/CD | **GitHub Actions** | lint, test, Docker build check, auto-deploy to HF Spaces on merge to `main` |

## Repo structure

```
sql/bureau_aggregation.sql        the bureau GROUP BY (see below)
src/credit_risk/
  data/                            DuckDB pipeline, leakage checks, synthetic-sample generator
  features/                        Polars feature engineering
  models/                          XGBoost + Optuna training, model artifact packaging
  evaluation/                      ROC-AUC/PR-AUC/Gini/KS/Brier, calibration, approval cutoffs
  explain/                         SHAP
  api/                             FastAPI backend (SOLID — see below)
notebooks/01_pd_model_pipeline.ipynb   the executed training pipeline
frontend/                          React + TypeScript client
infra/                             Dockerfile, docker-compose, MLflow server setup
scripts/                           local demo-model training, notebook generator
tests/                             20 tests: features, metrics, leakage checks, API smoke tests
data/sample/                       synthetic stand-in CSVs (schema-identical to the real dataset)
.github/workflows/ci.yml           lint, test, Docker build check, HF Spaces deploy
```

## Quickstart (local dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install ".[api,ml,dev]"

# generates data/sample/*.csv if not already present, trains a small model
# on it, and saves artifacts/model/{model.json,metadata.json}
python scripts/train_local_demo.py

ruff check src/ tests/ scripts/
pytest -q                          # 20 tests

# API
cp .env.example .env               # LOG_PREDICTIONS=false if you don't want to stand up Postgres
uvicorn credit_risk.api.main:app --reload

# frontend, in another shell
cd frontend && npm install
cp .env.example .env.local
npm run dev
```

Or with Docker (API + its prediction-logging Postgres):

```bash
python scripts/train_local_demo.py   # needs a model artifact to COPY into the image
docker compose -f infra/docker-compose.yml up --build
```

### Running the notebook

```bash
pip install ".[ml]" jupyter
jupyter lab notebooks/01_pd_model_pipeline.ipynb
```

## The bureau aggregation ([`sql/bureau_aggregation.sql`](sql/bureau_aggregation.sql))

`bureau.csv` is one-to-many per applicant — the aggregation is a `GROUP BY`
by nature, so it's SQL, run in DuckDB before anything touches Polars/pandas:

```sql
with prior_credit as (
    select
        SK_ID_CURR, SK_ID_BUREAU, CREDIT_ACTIVE, CREDIT_TYPE,
        DAYS_CREDIT, DAYS_CREDIT_ENDDATE, DAYS_ENDDATE_FACT, DAYS_CREDIT_UPDATE,
        CREDIT_DAY_OVERDUE, CNT_CREDIT_PROLONG,
        coalesce(AMT_CREDIT_SUM, 0)         as AMT_CREDIT_SUM,
        coalesce(AMT_CREDIT_SUM_DEBT, 0)    as AMT_CREDIT_SUM_DEBT,
        coalesce(AMT_CREDIT_SUM_LIMIT, 0)   as AMT_CREDIT_SUM_LIMIT,
        coalesce(AMT_CREDIT_SUM_OVERDUE, 0) as AMT_CREDIT_SUM_OVERDUE,
        coalesce(AMT_CREDIT_MAX_OVERDUE, 0) as AMT_CREDIT_MAX_OVERDUE
    from bureau
    where DAYS_CREDIT <= 0   -- leakage guard: drop any row dated after the application
)
select
    SK_ID_CURR,
    count(*)                                        as bureau_credit_count,
    count(distinct CREDIT_TYPE)                     as bureau_distinct_credit_types,
    count(*) filter (where CREDIT_ACTIVE = 'Active') as bureau_active_credit_count,
    avg(CREDIT_DAY_OVERDUE)                          as bureau_avg_days_overdue,
    sum(AMT_CREDIT_SUM)                              as bureau_total_credit_sum,
    sum(AMT_CREDIT_SUM_DEBT)                         as bureau_total_credit_debt,
    min(DAYS_CREDIT)                                 as bureau_days_since_oldest_credit,
    max(DAYS_CREDIT)                                 as bureau_days_since_most_recent_credit
    -- (see the full file for all ~18 aggregate columns)
from prior_credit
group by SK_ID_CURR
```

## Temporal leakage — the one thing verified personally

The most common, most dangerous bug class in credit-scoring pipelines:
confirming nothing derived from `bureau` postdates the loan application it's
joined onto. This isn't left to a linter or an AI-written assertion buried in
a pipeline — it's a standalone, unit-tested module
([`src/credit_risk/data/leakage_checks.py`](src/credit_risk/data/leakage_checks.py))
with two distinct checks:

1. **Hard leakage (structural, provable):** any `bureau` row with
   `DAYS_CREDIT > 0` is dated *after* the application. The SQL above filters
   these out; the Python module re-checks the raw data independently and
   **raises** if the assumption doesn't hold, rather than silently trusting
   the SQL filter. Also run explicitly, separately, in the notebook (section 2) 
   so the result is visible, not buried inside a pipeline call.

2. **Soft leakage (structural, not fully fixable at this scope):**
   `bureau.csv` is extracted at one fixed date for the *whole table*, not
   per application. `CREDIT_ACTIVE`/`AMT_CREDIT_SUM_DEBT` reflect status as
   of that extraction date, not each applicant's own application date — a
   credit that was open when someone applied may already read `"Closed"` if
   it closed before extraction. This can't be detected row-by-row from these
   two tables alone; it's reported as a quantified caveat (`stale_update_pct`
   in the leakage report), not swept under the rug.

## Evaluation
- **Stratified K-fold CV**
- **ROC-AUC, PR-AUC** — standard classification metrics
- **Gini coefficient, KS statistic** — industry-standard credit-scoring metrics
- **Calibration** (Brier score / reliability curve)

AI's try on the dataset
| Metric | Value |
|---|---|
| ROC-AUC | 0.7696 |
| PR-AUC | 0.2574 |
| Gini | 0.5392 |
| KS statistic | 0.4020 |
| Brier score | 0.1898 |

## API

`POST /predict` — see `src/credit_risk/api/schemas.py` for the full request
shape. `bureau_summary` is optional and defaults to "no record on file" (a
real, meaningful state for underbanked/thin-file applicants, not a
placeholder).

`GET /health` reports whether the model artifact loaded successfully.

## Backend design (SOLID)

`services/prediction_service.py` depends only on the `ModelRepository` and
`PredictionLogRepository` **interfaces** (`repositories/base.py`), not on
concrete Postgres/filesystem classes — dependency inversion, so the service
is unit-testable with fakes and the logging backend is swappable
(`NullPredictionLogRepository` is used automatically if `LOG_PREDICTIONS` is
off or Postgres is unreachable, so a logging outage never takes scoring
down with it). Wiring happens once, in `api/main.py`'s lifespan (the
composition root) — route handlers stay thin.

## Infra

- `infra/Dockerfile.api` — serving image (`api` + `serving-ml` extras only;
  training-only deps like `duckdb`/`optuna`/`mlflow` aren't in the served
  image)
- `infra/docker-compose.yml` — API + its prediction-logging Postgres (dev)
- `infra/mlflow-server/` — self-hosted MLflow (Postgres backend + MinIO
  artifact store, proxied-artifact mode) + Cloudflare Tunnel setup for
  exposing it to a Kaggle notebook — see that directory's README

## CI/CD

`.github/workflows/ci.yml`: lint (`ruff` + frontend `oxlint`/`tsc`), tests
(pytest — model loads, `/predict` returns a valid schema/probability range),
a Docker build check on every push/PR, and auto-deploy to HF Spaces on merge
to `main`. Failure notifications use GitHub Actions' own default
email-on-failure rather than a dedicated alerting service.

The deploy job stages a minimal HF-Spaces-shaped directory (root
`Dockerfile` + a `README.md` with the Spaces YAML config block — different
from this monorepo's layout) and pushes it via `huggingface_hub`.

It will refuse to run at all until `artifacts/model/model.json` and
`metadata.json` are committed to the repo (deliberately — deploying the
CI-only synthetic demo model to a "live" endpoint would be misleading; see
the job's first step).