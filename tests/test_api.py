from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

SAMPLE = {
    "mileage_km": 160000,
    "engine_temp_c": 105,
    "rpm": 2200,
    "vibration_level": 0.08,
    "oil_quality_pct": 55,
    "brake_pad_mm": 3.2,
    "battery_voltage": 12.1,
    "coolant_level_pct": 70,
    "vehicle_age_years": 11,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert len(response.json()["features"]) == 9


def test_predict_endpoint():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_failure_risk"] in [0, 1]
    assert 0 <= body["model_risk_score"] <= 1
    assert 0 <= body["vehicle_health_index"] <= 100
    assert "explainability" in body
    assert "recommendations" in body


def test_invalid_input_is_rejected():
    invalid = dict(SAMPLE)
    invalid["oil_quality_pct"] = 150
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_explanation_uses_shap_and_lists_features():
    body = client.post("/predict", json=SAMPLE).json()
    assert body["explainability"]["method"] == "SHAP TreeExplainer"
    assert len(body["explainability"]["top_features"]) == 5


def test_unknown_field_is_rejected():
    payload = dict(SAMPLE, surprise=1)
    assert client.post("/predict", json=payload).status_code == 422


def test_worn_vehicle_scores_higher_than_healthy_one():
    healthy = {
        "mileage_km": 30000, "engine_temp_c": 88, "rpm": 1500, "vibration_level": 0.03,
        "oil_quality_pct": 92, "brake_pad_mm": 9, "battery_voltage": 13.8,
        "coolant_level_pct": 90, "vehicle_age_years": 2,
    }
    worn = client.post("/predict", json=SAMPLE).json()["model_risk_score"]
    ok = client.post("/predict", json=healthy).json()["model_risk_score"]
    assert worn > ok
