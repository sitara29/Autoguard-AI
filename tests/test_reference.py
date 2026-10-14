"""Unit tests for the risk-index mapping and the shared maintenance rules."""
import numpy as np
import pandas as pd

from backend.reference import LOW_MAX, MODERATE_MAX, risk_band, risk_index
from backend.rules import RULES, triggered_rules


def test_risk_index_anchors_and_endpoints():
    t = 0.825
    assert risk_index(0.0, t) == 0.0
    assert risk_index(t, t) == 50.0
    assert risk_index(1.0, t) == 100.0


def test_risk_index_is_monotone_in_the_score():
    scores = np.linspace(0, 1, 501)
    idx = risk_index(scores, 0.825)
    assert np.all(np.diff(idx) >= 0)


def test_band_boundaries():
    assert risk_band(LOW_MAX - 0.01) == "LOW"
    assert risk_band(LOW_MAX) == "MODERATE"
    assert risk_band(MODERATE_MAX - 0.01) == "MODERATE"
    assert risk_band(MODERATE_MAX) == "HIGH"


def test_decision_threshold_is_the_start_of_the_high_band():
    for t in (0.5, 0.825, 0.9):
        assert risk_band(risk_index(t, t)) == "HIGH"
        assert risk_band(risk_index(t - 1e-6, t)) != "HIGH"


def test_rules_work_on_dicts_and_dataframes_identically():
    rows = [
        dict(mileage_km=30000, engine_temp_c=88, rpm=1500, vibration_level=0.03, oil_quality_pct=92,
             brake_pad_mm=9, battery_voltage=13.8, coolant_level_pct=90, vehicle_age_years=2),
        dict(mileage_km=200000, engine_temp_c=115, rpm=2500, vibration_level=0.05, oil_quality_pct=40,
             brake_pad_mm=2.0, battery_voltage=11.5, coolant_level_pct=50, vehicle_age_years=14),
    ]
    frame = pd.DataFrame(rows)
    for rule in RULES:
        vectorised = np.asarray(rule["test"](frame), dtype=bool)
        single = [bool(rule["test"](r)) for r in rows]
        assert list(vectorised) == single
    assert triggered_rules(rows[0]) == []
    assert {r["id"] for r in triggered_rules(rows[1])} == {r["id"] for r in RULES}
