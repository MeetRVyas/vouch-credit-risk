import pytest
from fastapi.testclient import TestClient

from credit_risk.api.config import Settings
from credit_risk.api.main import create_app


@pytest.fixture()
def test_settings() -> Settings:
    return Settings(
        model_artifact_dir="artifacts/model",
        log_predictions=False,  # smoke tests shouldn't require a live Postgres
    )


@pytest.fixture()
def client(test_settings: Settings):
    app = create_app(test_settings)
    with TestClient(app) as c:
        yield c
