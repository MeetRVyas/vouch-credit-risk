# MLflow Tracking Server

Self-hosted, not the Kaggle-local default -- so tuning runs survive past
Kaggle's ephemeral session and the full run history (params, metrics, model +
SHAP artifacts) lives somewhere durable.

**Stack:** Postgres (backend store, run/param/metric metadata) + MinIO
(artifact store, S3-compatible) + the `mlflow server` process, all via
`docker-compose.yml` in this directory. Separate from the API's
prediction-logging Postgres (`../docker-compose.yml`) -- different lifecycle,
different consumer.

## 1. Start the stack

```bash
docker compose -f infra/mlflow-server/docker-compose.yml up --build -d
```

This brings up:
- `mlflow-db` (Postgres, backend store)
- `minio` (S3-compatible artifact store) + a one-shot `create-bucket` job
- `mlflow-server` on port 5000, in **server-side proxied artifact mode**
  (`--serve-artifacts`) -- clients talk to the MLflow server over HTTP only;
  they never need raw MinIO/S3 credentials, just `MLFLOW_TRACKING_URI`.

Check it's up: open `http://localhost:5000` (or the tunnel URL, once step 2
is done).

## 2. Expose it to the Kaggle notebook

Kaggle notebooks can't reach `localhost:5000` on your machine directly. Use a
Cloudflare Tunnel (free, no port-forwarding, no static IP needed):

```bash
# one-time: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
cloudflared tunnel --url http://localhost:5000
```

This prints a `https://<random-subdomain>.trycloudflare.com` URL. That's your
`MLFLOW_TRACKING_URI` for the Kaggle session.

> If a training run is going to span multiple days, a quick tunnel isn't
> durable enough (it dies when the `cloudflared` process does) -- use a
> small VPS running MLflow instead, or a named/persistent Cloudflare Tunnel
> tied to a domain you control.

## 3. Point the Kaggle notebook at it

In the Kaggle notebook (see `notebooks/01_pd_model_pipeline.ipynb`), the only
secret needed is the tracking URI -- add it as a Kaggle Secret, don't hardcode
it:

```python
import mlflow
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])  # from Kaggle Secrets
mlflow.set_experiment("credit-default-risk")
mlflow.xgboost.autolog()
```

That's it -- no `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` needed on the
Kaggle side, because artifacts are proxied through the MLflow server rather
than uploaded directly to MinIO from the client.

## 4. After training: export for the repo

The real `mlruns/` (or an exported run-comparison CSV) is a "from developer
after training" deliverable, not something committed by the AI-assisted
scaffold. From the Kaggle notebook or a machine with `MLFLOW_TRACKING_URI`
set:

```bash
# either copy the tracking-server's Postgres+MinIO data directly, or export
# a flat comparison CSV of all trials:
python - <<'PY'
import mlflow
runs = mlflow.search_runs(experiment_names=["credit-default-risk"])
runs.to_csv("mlruns_export/trials.csv", index=False)
PY
```

## Local credentials (dev only)

`docker-compose.yml` in this directory uses fixed dev credentials
(`mlflow`/`mlflow`, `minioadmin`/`minioadmin`) suitable for a local/tunnel
setup only. See `.env.example` for the pattern to use if you move this to a
real VPS instead of a quick tunnel -- don't reuse these credentials anywhere
public-facing.
