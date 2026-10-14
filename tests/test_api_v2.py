"""Tests for the extended API: full SHAP output, component status, risk bands,
maintenance priority, demo samples and the metadata / model-info endpoints."""
import math

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)
COMPONENT_IDS = {"engine", "cooling", "lubrication", "brakes", "battery", "general_wear"}
FEATURES = {
    "mileage_km", "engine_temp_c", "rpm", "vibration_level", "oil_quality_pct",
    "brake_pad_mm", "battery_voltage", "coolant_level_pct", "vehicle_age_years",
}
OLD_SAMPLE = {
    "mileage_km": 160000, "engine_temp_c": 105, "rpm": 2200, "vibration_level": 0.08,
    "oil_quality_pct": 55, "brake_pad_mm": 3.2, "battery_voltage": 12.1,
    "coolant_level_pct": 70, "vehicle_age_years": 11,
}


def demo_samples() -> dict:
    body = client.get("/samples").json()
    return {s["id"]: s for s in body["samples"]}


def predict(inputs: dict) -> dict:
    response = client.post("/predict", json=inputs)
    assert response.status_code == 200, response.text
    return response.json()


def test_explanation_lists_all_nine_features_sorted_by_impact():
    for sample in demo_samples().values():
        explain = predict(sample["inputs"])["explainability"]
        rows = explain["contributions"]
        assert {r["feature"] for r in rows} == FEATURES
        magnitudes = [abs(r["contribution"]) for r in rows]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert explain["top_features"] == rows[:5]


def test_shap_contributions_add_up_to_the_model_score():
    """SHAP additivity: base value + contributions = the model's raw log-odds output."""
    for sample in demo_samples().values():
        body = predict(sample["inputs"])
        explain = body["explainability"]
        total = explain["base_value"] + sum(r["contribution"] for r in explain["contributions"])
        score = body["model_risk_score"]
        logit = math.log(score / (1 - score))
        assert abs(total - explain["raw_margin"]) < 5e-3
        assert abs(logit - explain["raw_margin"]) < 5e-3


def test_components_cover_every_feature_exactly_once():
    body = predict(OLD_SAMPLE)
    comps = body["components"]
    assert {c["id"] for c in comps} == COMPONENT_IDS
    assert all(c["status"] in {"ok", "watch", "critical"} for c in comps)
    listed = [f["feature"] for c in comps for f in c["features"]]
    assert sorted(listed) == sorted(FEATURES)


def test_class_band_and_risk_index_agree():
    for sample in demo_samples().values():
        body = predict(sample["inputs"])
        is_high = body["predicted_failure_risk"] == 1
        assert is_high == (body["risk_level"] == "HIGH") == (body["risk_index"] >= 50)
        assert 0 <= body["risk_index"] <= 100


def test_health_index_is_derived_from_risk_index():
    for sample in demo_samples().values():
        body = predict(sample["inputs"])
        assert abs(body["vehicle_health_index"] - (100 - body["risk_index"])) <= 1


def test_demo_vehicles_land_in_their_advertised_bands():
    samples = demo_samples()
    assert predict(samples["healthy"]["inputs"])["risk_level"] == "LOW"
    assert predict(samples["moderate"]["inputs"])["risk_level"] == "MODERATE"
    assert predict(samples["high_risk"]["inputs"])["risk_level"] == "HIGH"


def test_samples_are_real_rows_with_dataset_labels():
    for sample in demo_samples().values():
        assert set(sample["inputs"]) == FEATURES
        assert sample["dataset_label"] in (0, 1)
        assert "held-out" in sample["description"]


def test_moderate_band_without_high_rule_is_not_priority_high():
    body = predict(demo_samples()["moderate"]["inputs"])
    high_rules = [r for r in body["recommendations"]
                  if r["source"] == "rule_based_sensor_advisory" and r["severity"] == "high"]
    assert body["predicted_failure_risk"] == 0 and not high_rules
    assert body["maintenance_priority"] == "MEDIUM"


def test_high_severity_rule_forces_high_priority_even_in_low_band():
    inputs = dict(demo_samples()["healthy"]["inputs"], engine_temp_c=115)
    body = predict(inputs)
    assert "engine_overheat" in {r["id"] for r in body["recommendations"]}
    assert body["maintenance_priority"] == "HIGH"


def test_healthy_demo_vehicle_has_low_priority_and_no_alarms():
    body = predict(demo_samples()["healthy"]["inputs"])
    assert body["maintenance_priority"] == "LOW"
    assert {c["status"] for c in body["components"]} == {"ok"}
    assert [r["id"] for r in body["recommendations"]] == ["no_advisory"]


def test_rule_recommendations_carry_dataset_evidence():
    body = predict(OLD_SAMPLE)
    rules = [r for r in body["recommendations"] if r["source"] == "rule_based_sensor_advisory"
             and r["id"] != "no_advisory"]
    assert rules
    for r in rules:
        assert r["evidence"]["strength"] in {"strong", "moderate", "weak", "none"}
        assert r["component"] in COMPONENT_IDS


def test_input_outside_training_range_is_flagged():
    base = demo_samples()["healthy"]["inputs"]
    assert predict(base)["input_warnings"] == []
    warned = predict(dict(base, vibration_level=6.0))["input_warnings"]
    assert [w["feature"] for w in warned] == ["vibration_level"]


def test_metadata_exposes_ranges_components_bands_and_rules():
    meta = client.get("/metadata").json()
    assert {f["name"] for f in meta["features"]} == FEATURES
    for f in meta["features"]:
        assert f["unit"] and f["min"] < f["max"] and f["component"] in COMPONENT_IDS
    assert {c["id"] for c in meta["components"]} == COMPONENT_IDS
    assert {"LOW", "MODERATE", "HIGH"} <= set(meta["risk_bands"])
    assert len(meta["maintenance_rules"]) == 6
    assert 0 < meta["decision_threshold"] < 1


def test_model_info_reports_comparison_tuning_and_test_results():
    info = client.get("/model-info").json()
    assert set(info["tuned_5_fold_cv"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert set(info["baseline_5_fold_cv"]) == set(info["tuned_5_fold_cv"])
    assert info["model_selection"]["selected_model"] == "XGBoost"
    assert info["split"]["stratified"] is True
    assert info["caveats"]


def test_cors_preflight_allows_the_react_dev_server():
    response = client.options(
        "/predict",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
