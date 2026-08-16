def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_predict_returns_valid_schema_and_probability_range(client):
    payload = {
        "applicant": {
            "name_contract_type": "Cash loans",
            "code_gender": "F",
            "flag_own_car": "N",
            "flag_own_realty": "Y",
            "cnt_children": 1,
            "cnt_fam_members": 3,
            "amt_income_total": 135000,
            "amt_credit": 450000,
            "amt_annuity": 27000,
            "name_income_type": "Working",
            "name_education_type": "Higher education",
            "name_family_status": "Married",
            "name_housing_type": "House / apartment",
            "age_years": 34,
            "years_employed": 6,
            "region_rating_client": 2,
            "ext_source_2": 0.55,
        },
        "bureau_summary": {
            "bureau_credit_count": 3,
            "bureau_active_credit_count": 1,
            "bureau_total_credit_sum": 80000,
            "bureau_total_credit_debt": 20000,
        },
        "reference_id": "test-ref-123",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert 0.0 <= body["predicted_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["model_version"]
    assert body["reference_id"] == "test-ref-123"
    assert len(body["top_risk_factors"]) == 5
    for factor in body["top_risk_factors"]:
        assert factor["direction"] in {"increases_risk", "decreases_risk"}
        assert isinstance(factor["feature"], str)


def test_predict_works_with_no_bureau_history_supplied(client):
    """bureau_summary is optional -- a thin-file applicant should still score."""
    payload = {
        "applicant": {
            "amt_income_total": 90000,
            "amt_credit": 200000,
            "amt_annuity": 15000,
            "age_years": 25,
        }
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["predicted_probability"] <= 1.0


def test_predict_rejects_negative_income(client):
    payload = {
        "applicant": {
            "amt_income_total": -1000,
            "amt_credit": 200000,
            "amt_annuity": 15000,
            "age_years": 25,
        }
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_age_below_minimum(client):
    payload = {
        "applicant": {
            "amt_income_total": 90000,
            "amt_credit": 200000,
            "amt_annuity": 15000,
            "age_years": 12,
        }
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
