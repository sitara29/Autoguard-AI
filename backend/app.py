"""FastAPI backend for AI-Powered Vehicle Failure Risk Prediction.

Run from the project root with:
    uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000

Endpoints
    GET  /            service info
    GET  /health      model/service health
    GET  /metadata    features (labels, units, training ranges), components, risk bands, rules
    GET  /samples     real dataset rows to use as demo vehicles
    GET  /model-info  model comparison, tuning, cross-validation and test results
    POST /predict     risk prediction + SHAP explanation + component status + maintenance advice
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from backend.reference import risk_band, risk_index
from backend.rules import RULES, SEVERITY_ORDER, triggered_rules

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "backend" / "models" / "vehicle_failure_model.joblib"
REPORT_PATH = ROOT / "reports" / "model_metrics.json"
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Trained model not found at {MODEL_PATH}. Run `python train_model.py` from the project root."
    )

BUNDLE: dict[str, Any] = joblib.load(MODEL_PATH)
if "reference" not in BUNDLE:
    raise RuntimeError(
        "The saved model has no reference data (feature ranges, SHAP grading, risk bands). "
        "Re-run `python train_model.py` to regenerate it."
    )
REF: dict[str, Any] = BUNDLE["reference"]
PIPELINE = BUNDLE["pipeline"]
THRESHOLD = float(BUNDLE["threshold"])
FEATURES: list[str] = list(BUNDLE["features"])
TARGET = BUNDLE.get("target", "failure_risk")
MODEL = PIPELINE.named_steps["model"]
IMPUTER = PIPELINE.named_steps.get("imputer")
SCORE_NOTE = BUNDLE.get(
    "score_note", "Uncalibrated model score, not a verified real-world failure probability."
)

app = FastAPI(
    title="Vehicle Failure Risk Prediction API",
    description=(
        "Predicts the dataset's binary failure-risk label from nine vehicle parameters, "
        "explains it with local SHAP contributions, and returns rule-based maintenance advisories."
    ),
    version="2.0.0",
)

origins_env = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class VehicleInput(BaseModel):
    """Vehicle inputs. Limits are broad data-entry sanity checks, not failure thresholds."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mileage_km: float = Field(ge=0, le=1_000_000, description="Odometer reading in kilometres")
    engine_temp_c: float = Field(ge=-50, le=250, description="Engine temperature in Celsius")
    rpm: float = Field(ge=0, le=20_000, description="Engine revolutions per minute")
    vibration_level: float = Field(ge=0, le=10, description="Vibration sensor level (training data range is much narrower; see /metadata)")
    oil_quality_pct: float = Field(ge=0, le=100, description="Oil-quality index, percent")
    brake_pad_mm: float = Field(ge=0, le=50, description="Remaining brake-pad thickness, millimetres")
    battery_voltage: float = Field(ge=0, le=60, description="Battery voltage")
    coolant_level_pct: float = Field(ge=0, le=100, description="Coolant level, percent")
    vehicle_age_years: float = Field(ge=0, le=100, description="Vehicle age in years")


@lru_cache(maxsize=1)
def get_explainer():
    """Create the SHAP tree explainer lazily so API startup is quick."""
    return shap.TreeExplainer(MODEL)


@lru_cache(maxsize=1)
def get_report() -> dict[str, Any] | None:
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text())
    return None


# --------------------------------------------------------------------------- helpers
def contribution_level(value: float, cutoffs: dict[str, float]) -> str:
    """Grade a SHAP contribution (log-odds). NONE = does not push toward higher risk."""
    if value <= 0:
        return "NONE"
    if value >= cutoffs["high"]:
        return "HIGH"
    if value >= cutoffs["medium"]:
        return "MEDIUM"
    return "LOW"


def input_percentile(name: str, value: float) -> float:
    """Where this input sits in the training distribution (0-100)."""
    grid = REF["features"][name]["percentile_grid"]
    return round(float(np.interp(value, grid["values"], grid["percentiles"])), 1)


def input_warnings(values: dict[str, float]) -> list[dict[str, Any]]:
    """Flag inputs outside the range the model was trained on."""
    warnings: list[dict[str, Any]] = []
    for name in FEATURES:
        ref = REF["features"][name]
        value = values[name]
        if value < ref["min"] or value > ref["max"]:
            warnings.append(
                {
                    "feature": name,
                    "label": ref["label"],
                    "value": value,
                    "training_min": ref["min"],
                    "training_max": ref["max"],
                    "message": (
                        f"{ref['label']} of {value:g} {ref['unit']} is outside the range seen in the "
                        f"training data ({ref['min']:g}-{ref['max']:g}); the prediction for this input "
                        "may be unreliable."
                    ),
                }
            )
    return warnings


def explain_prediction(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, float] | None]:
    """Local SHAP contributions for all features (class 1 = higher-risk label)."""
    try:
        transformed = IMPUTER.transform(frame) if IMPUTER is not None else frame.to_numpy()
        values = np.asarray(get_explainer().shap_values(transformed))
        if values.ndim == 3:
            values = values[..., -1]
        raw = values[0]
        by_feature = {name: float(raw[i]) for i, name in enumerate(FEATURES)}
        contributions = []
        for name in FEATURES:
            ref = REF["features"][name]
            c = by_feature[name]
            contributions.append(
                {
                    "feature": name,
                    "label": ref["label"],
                    "unit": ref["unit"],
                    "component": ref["component"],
                    "input_value": float(frame.iloc[0][name]),
                    "input_percentile": input_percentile(name, float(frame.iloc[0][name])),
                    "contribution": round(c, 6),
                    "direction": "pushes_toward_higher_risk" if c > 0 else "pushes_toward_lower_risk",
                    "level": contribution_level(c, ref["shap_levels"]),
                }
            )
        contributions.sort(key=lambda row: abs(row["contribution"]), reverse=True)
        base_value = float(REF["base_value"])
        return (
            {
                "method": "SHAP TreeExplainer",
                "contribution_scale": "model raw score (log-odds), not probability points",
                "base_value": round(base_value, 6),
                "raw_margin": round(base_value + sum(by_feature.values()), 6),
                "level_rule": REF["contribution_grading"]["rule"],
                "contributions": contributions,
                "top_features": contributions[:5],
                "note": "Contributions explain this model output; they are not causal proof or a mechanical diagnosis. "
                        "Correlated inputs (mileage, age, brake wear, oil quality) share credit, so a low contribution "
                        "for one of them does not mean it is unimportant.",
            },
            by_feature,
        )
    except Exception as exc:  # Prediction should remain available if explanation fails.
        return (
            {
                "method": "SHAP unavailable for this request",
                "contribution_scale": None,
                "contributions": [],
                "top_features": [],
                "note": f"Prediction succeeded, but local explanation failed ({type(exc).__name__}).",
            },
            None,
        )


def rule_evidence_note(rule_id: str) -> tuple[dict[str, Any], str]:
    ev = REF["rule_evidence"][rule_id]
    rate, base, lift = ev["failure_rate_when_triggered"], ev["baseline_failure_rate"], ev["lift"]
    if rate is None or lift is None:
        return ev, "No training vehicles met this condition, so there is no dataset evidence for it."
    text = (
        f"In the training data, {ev['share_of_vehicles'] * 100:.1f}% of vehicles met this condition and "
        f"{rate * 100:.1f}% of them carried the high-risk label (vs {base * 100:.1f}% overall, {lift:.1f}x)."
    )
    if ev["strength"] == "weak":
        text += " The association is weak in this dataset; the advisory reflects general maintenance practice."
    return ev, text


def make_recommendations(
    values: dict[str, float], predicted_class: int, level: str
) -> list[dict[str, Any]]:
    """Transparent advisories: model output plus rule-based sensor checks."""
    recs: list[dict[str, Any]] = []
    if predicted_class == 1:
        recs.append(
            {
                "id": "model_high_risk",
                "severity": "high",
                "system": "vehicle",
                "component": "vehicle",
                "message": "The model classified this input as higher failure risk. Arrange a professional vehicle inspection soon.",
                "source": "model_output",
            }
        )
    elif level == "MODERATE":
        recs.append(
            {
                "id": "model_moderate_risk",
                "severity": "medium",
                "system": "vehicle",
                "component": "vehicle",
                "message": "The model places this input in the moderate-risk band. Consider scheduling a routine inspection.",
                "source": "model_output",
            }
        )
    for rule in triggered_rules(values):
        evidence, note = rule_evidence_note(rule["id"])
        recs.append(
            {
                "id": rule["id"],
                "severity": rule["severity"],
                "system": rule["system"],
                "component": rule["component"],
                "message": rule["message"],
                "source": "rule_based_sensor_advisory",
                "condition": rule["condition"],
                "evidence": {
                    "strength": evidence["strength"],
                    "lift": evidence["lift"],
                    "failure_rate_when_triggered": evidence["failure_rate_when_triggered"],
                    "baseline_failure_rate": evidence["baseline_failure_rate"],
                    "n_triggered_in_training": evidence["n_triggered"],
                    "note": note,
                },
            }
        )
    recs.sort(key=lambda r: SEVERITY_ORDER[r["severity"]])
    if not recs:
        recs.append(
            {
                "id": "no_advisory",
                "severity": "info",
                "system": "general_maintenance",
                "component": "vehicle",
                "message": "No rule-based advisory was triggered by these input checks. Continue manufacturer-recommended scheduled maintenance.",
                "source": "rule_based_sensor_advisory",
            }
        )
    return recs


def component_summaries(
    by_feature: dict[str, float] | None,
    contributions: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-component status for the 3D hotspots: combines SHAP grading with triggered rules."""
    by_name = {row["feature"]: row for row in contributions}
    out: list[dict[str, Any]] = []
    for cid, info in REF["components"].items():
        members = info["features"]
        rules = [r for r in recommendations if r["component"] == cid and r["source"] == "rule_based_sensor_advisory"]
        if by_feature is not None:
            total = sum(by_feature[f] for f in members)
            level = contribution_level(total, info["shap_levels"])
            ai = {"value": round(total, 6), "level": level}
        else:
            level, ai = "UNKNOWN", {"value": None, "level": "UNKNOWN"}
        severities = {r["severity"] for r in rules}
        if level == "HIGH" or "high" in severities:
            status = "critical"
        elif level == "MEDIUM" or "medium" in severities:
            status = "watch"
        else:
            status = "ok"
        out.append(
            {
                "id": cid,
                "label": info["label"],
                "status": status,
                "ai_contribution": ai,
                "features": [by_name[f] for f in members if f in by_name],
                "advisories": [r["id"] for r in rules],
            }
        )
    return out


def maintenance_priority(
    predicted_class: int,
    level: str,
    components: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> str:
    """Overall priority from the model's decision and the transparent rules.

    HIGH   : the model flags the vehicle (class 1) or a high-severity sensor rule fired.
    MEDIUM : moderate-risk band, a medium-severity rule fired, or SHAP marks a component
             as a notable contributor. SHAP grading alone never raises priority to HIGH.
    LOW    : otherwise.
    """
    rules = [r for r in recommendations if r["source"] == "rule_based_sensor_advisory"]
    if predicted_class == 1 or any(r["severity"] == "high" for r in rules):
        return "HIGH"
    if (
        level == "MODERATE"
        or any(r["severity"] == "medium" for r in rules)
        or any(c["status"] in ("critical", "watch") for c in components)
    ):
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------- routes
@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Vehicle Failure Risk Prediction API",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/health", "/metadata", "/samples", "/model-info", "/predict"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": BUNDLE.get("model_name", "unknown"),
        "feature_count": len(FEATURES),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "features": [
            {"name": name, **{k: v for k, v in REF["features"][name].items()
                              if k not in ("percentile_grid", "shap_levels")}}
            for name in FEATURES
        ],
        "components": [
            {"id": cid, "label": info["label"], "features": info["features"]}
            for cid, info in REF["components"].items()
        ],
        "target": TARGET,
        "model_name": BUNDLE.get("model_name", "unknown"),
        "class_1_meaning": "higher risk according to the dataset's failure_risk label",
        "decision_threshold": THRESHOLD,
        "risk_bands": {
            "index": "0-100 display index, monotone in the model score; the decision threshold maps to 50",
            "LOW": f"index < {REF['risk_index']['low_max']:g}",
            "MODERATE": f"{REF['risk_index']['low_max']:g} <= index < {REF['risk_index']['moderate_max']:g}",
            "HIGH": f"index >= {REF['risk_index']['moderate_max']:g} (same as predicted class 1)",
            "dataset_failure_rate_by_band": REF["band_evidence"],
            "basis": REF["band_evidence_basis"],
        },
        "maintenance_rules": [
            {
                "id": rule["id"],
                "component": rule["component"],
                "severity": rule["severity"],
                "condition": rule["condition"],
                "evidence": REF["rule_evidence"][rule["id"]],
            }
            for rule in RULES
        ],
        "score_note": SCORE_NOTE,
        "frontend_notes": {
            "vehicle_health_index": "100 minus risk_index; a display index, not a separately trained or validated health metric",
            "risk_score": "model_risk_score is an uncalibrated score inflated by class weighting; show risk_level / risk_index instead of the raw percentage",
            "vibration_level": "units are not specified in the dataset; the training range is narrow, so use input_warnings and the ranges above for sliders",
        },
    }


@app.get("/samples")
def samples() -> dict[str, Any]:
    return {
        "note": "Real rows from the held-out test split; dataset_label is the dataset's failure_risk value.",
        "samples": REF["samples"],
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    report = get_report()
    if report is None:
        raise HTTPException(status_code=404, detail="reports/model_metrics.json not found. Run train_model.py.")
    return report


@app.post("/predict")
def predict(payload: VehicleInput) -> dict[str, Any]:
    values = payload.model_dump()
    frame = pd.DataFrame([{feature: values[feature] for feature in FEATURES}], columns=FEATURES)

    try:
        model_score = float(PIPELINE.predict_proba(frame)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed. Check backend model compatibility.") from exc

    predicted_class = int(model_score >= THRESHOLD)
    index = risk_index(model_score, THRESHOLD)
    level = risk_band(index)
    health_index = int(max(0, min(100, round(100 - index))))

    explainability, by_feature = explain_prediction(frame)
    recommendations = make_recommendations(values, predicted_class, level)
    components = component_summaries(by_feature, explainability["contributions"], recommendations)
    band_ref = REF["band_evidence"][level]

    return {
        "predicted_failure_risk": predicted_class,
        "risk_label": "HIGHER_RISK" if predicted_class == 1 else "LOWER_RISK",
        "risk_level": level,
        "risk_index": round(index, 1),
        "model_risk_score": round(model_score, 6),
        "model_risk_score_percent": round(model_score * 100, 1),
        "decision_threshold": THRESHOLD,
        "vehicle_health_index": health_index,
        "health_index_note": "Derived display index (100 minus risk_index); not a separately trained health target.",
        "band_context": {
            "band": level,
            "dataset_failure_rate": band_ref["dataset_failure_rate"],
            "n_vehicles": band_ref["n_vehicles"],
            "basis": REF["band_evidence_basis"],
        },
        "maintenance_priority": maintenance_priority(predicted_class, level, components, recommendations),
        "input_features": values,
        "input_warnings": input_warnings(values),
        "explainability": explainability,
        "components": components,
        "recommendations": recommendations,
        "disclaimer": (
            "Research prototype. The model classifies the dataset's synthetic failure_risk label; "
            "it does not predict a time or date to failure and does not replace a qualified inspection."
        ),
    }
