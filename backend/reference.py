"""Reference data the API needs besides the fitted model.

Built once at training time (training partition only, plus demo rows taken from
the held-out split) and stored inside the model bundle, so the API can:
  * show feature ranges / gauge positions from the real training data,
  * grade SHAP contributions on a stable scale,
  * convert the model score into a threshold-anchored risk index and band,
  * attach dataset evidence to each maintenance rule,
  * offer real dataset rows as demo vehicles.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from backend.rules import RULES

FEATURE_META: dict[str, dict[str, str]] = {
    "mileage_km": {"label": "Mileage", "unit": "km", "component": "general_wear"},
    "engine_temp_c": {"label": "Engine temperature", "unit": "°C", "component": "engine"},
    "rpm": {"label": "Engine RPM", "unit": "rpm", "component": "engine"},
    "vibration_level": {"label": "Vibration level", "unit": "index", "component": "engine"},
    "oil_quality_pct": {"label": "Oil quality", "unit": "%", "component": "lubrication"},
    "brake_pad_mm": {"label": "Brake pad thickness", "unit": "mm", "component": "brakes"},
    "battery_voltage": {"label": "Battery voltage", "unit": "V", "component": "battery"},
    "coolant_level_pct": {"label": "Coolant level", "unit": "%", "component": "cooling"},
    "vehicle_age_years": {"label": "Vehicle age", "unit": "years", "component": "general_wear"},
}

COMPONENTS: dict[str, str] = {
    "engine": "Engine",
    "cooling": "Cooling system",
    "lubrication": "Oil / lubrication",
    "brakes": "Brakes",
    "battery": "Battery",
    "general_wear": "Overall wear (age & mileage)",
}

# Risk index: 0-100 display scale, monotone in the model score and anchored so
# that the tuned decision threshold sits exactly at 50. Index >= 50 <=> class 1.
LOW_MAX = 30.0       # index < 30        -> LOW
MODERATE_MAX = 50.0  # 30 <= index < 50  -> MODERATE; >= 50 -> HIGH

# Contribution grading (SHAP, log-odds scale): level cut-offs are the larger of a
# population percentile and an absolute floor, so tiny pushes are never "HIGH".
MEDIUM_FLOOR, HIGH_FLOOR = 0.10, 0.25
MEDIUM_PCT, HIGH_PCT = 75, 90


def risk_index(score: float | np.ndarray, threshold: float):
    """Map a model score in [0, 1] to a 0-100 index with `threshold` -> 50."""
    s = np.asarray(score, dtype=float)
    idx = np.where(
        s < threshold,
        50.0 * s / threshold,
        50.0 + 50.0 * (s - threshold) / (1.0 - threshold),
    )
    return float(idx) if idx.ndim == 0 else idx


def risk_band(index: float | np.ndarray):
    i = np.asarray(index, dtype=float)
    out = np.where(i < LOW_MAX, "LOW", np.where(i < MODERATE_MAX, "MODERATE", "HIGH"))
    return str(out) if out.ndim == 0 else out


def _cutoffs(values: np.ndarray) -> dict[str, float]:
    return {
        "medium": float(max(np.percentile(values, MEDIUM_PCT), MEDIUM_FLOOR)),
        "high": float(max(np.percentile(values, HIGH_PCT), HIGH_FLOOR)),
    }


def _nearest_to_median(series: pd.Series) -> Any:
    """Index label of the entry closest to the median of `series`."""
    return (series - series.median()).abs().idxmin()


def build_reference(
    pipeline,
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    oof_scores: np.ndarray,
    threshold: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_train = np.asarray(explainer.shap_values(imputer.transform(X_train)))
    if shap_train.ndim == 3:  # some SHAP versions return (rows, features, classes)
        shap_train = shap_train[..., -1]
    shap_df = pd.DataFrame(shap_train, columns=features)
    base_value = float(np.ravel(explainer.expected_value)[0])

    # Per-feature statistics from the training partition.
    pct_grid = np.linspace(0, 100, 21)
    feature_ref: dict[str, Any] = {}
    for name in features:
        col = X_train[name].astype(float)
        feature_ref[name] = {
            **FEATURE_META[name],
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "p5": float(col.quantile(0.05)),
            "p25": float(col.quantile(0.25)),
            "p50": float(col.quantile(0.50)),
            "p75": float(col.quantile(0.75)),
            "p95": float(col.quantile(0.95)),
            "percentile_grid": {
                "percentiles": pct_grid.tolist(),
                "values": [float(np.percentile(col, p)) for p in pct_grid],
            },
            "shap_levels": _cutoffs(shap_df[name].to_numpy()),
        }

    # Component grouping and its SHAP grading.
    components: dict[str, Any] = {}
    for cid, label in COMPONENTS.items():
        members = [f for f in features if FEATURE_META[f]["component"] == cid]
        components[cid] = {
            "label": label,
            "features": members,
            "shap_levels": _cutoffs(shap_df[members].sum(axis=1).to_numpy()),
        }

    # Failure rate of the dataset label inside each risk band, using out-of-fold
    # training predictions only (no test labels involved).
    oof_band = risk_band(risk_index(oof_scores, threshold))
    y_arr = np.asarray(y_train)
    band_evidence = {}
    for band in ("LOW", "MODERATE", "HIGH"):
        mask = oof_band == band
        band_evidence[band] = {
            "n_vehicles": int(mask.sum()),
            "dataset_failure_rate": float(y_arr[mask].mean()) if mask.any() else None,
        }

    # Association of each maintenance rule with the dataset label (training data).
    baseline_rate = float(y_train.mean())
    rule_evidence: dict[str, Any] = {}
    for rule in RULES:
        mask = np.asarray(rule["test"](X_train), dtype=bool)
        rate = float(y_arr[mask].mean()) if mask.any() else None
        lift = (rate / baseline_rate) if rate is not None and baseline_rate else None
        strength = (
            "none" if lift is None else "strong" if lift >= 3 else "moderate" if lift >= 1.5 else "weak"
        )
        rule_evidence[rule["id"]] = {
            "n_triggered": int(mask.sum()),
            "share_of_vehicles": float(mask.mean()),
            "failure_rate_when_triggered": rate,
            "baseline_failure_rate": baseline_rate,
            "lift": lift,
            "strength": strength,
        }

    # Demo vehicles: real rows from the held-out split (never used for training).
    scores_test = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=X_test.index)
    idx_test = pd.Series(risk_index(scores_test.to_numpy(), threshold), index=X_test.index)
    band_test = pd.Series(risk_band(idx_test.to_numpy()), index=X_test.index)
    samples: list[dict[str, Any]] = []

    def add_sample(sample_id: str, name: str, row_label, blurb: str) -> None:
        samples.append(
            {
                "id": sample_id,
                "name": name,
                "description": f"{blurb} Real row #{int(row_label)} from the held-out test split.",
                "dataset_label": int(y_test.loc[row_label]),
                "source_row": int(row_label),
                "inputs": {f: float(X_test.loc[row_label, f]) for f in features},
            }
        )

    low = band_test[(band_test == "LOW") & (y_test == 0)].index
    if len(low):
        add_sample("healthy", "Healthy vehicle", _nearest_to_median(scores_test[low]),
                   "Typical low-risk vehicle.")
    mod = band_test[band_test == "MODERATE"].index
    if len(mod):
        add_sample("moderate", "Needs watching", _nearest_to_median(scores_test[mod]),
                   "Moderate-risk vehicle.")
    high = band_test[(band_test == "HIGH") & (y_test == 1)].index
    if len(high):
        add_sample("high_risk", "High-risk vehicle", _nearest_to_median(scores_test[high]),
                   "Vehicle the model flags as high risk, and the dataset labels high risk too.")
    hot = X_test.index[(X_test["engine_temp_c"] >= 110) & (X_test["brake_pad_mm"] >= 4)]
    if len(hot):
        add_sample("overheating", "Overheating engine", _nearest_to_median(scores_test[hot]),
                   "Engine temperature above the 110 °C advisory level while the brakes are fine.")

    return {
        "base_value": base_value,
        "features": feature_ref,
        "components": components,
        "risk_index": {
            "anchor": "decision threshold maps to index 50",
            "low_max": LOW_MAX,
            "moderate_max": MODERATE_MAX,
            "threshold": float(threshold),
        },
        "contribution_grading": {
            "scale": "SHAP log-odds",
            "rule": f"HIGH >= max(P{HIGH_PCT} of training contributions, {HIGH_FLOOR}); "
                    f"MEDIUM >= max(P{MEDIUM_PCT}, {MEDIUM_FLOOR}); LOW > 0; NONE <= 0",
        },
        "band_evidence": band_evidence,
        "band_evidence_basis": "out-of-fold predictions on the training partition",
        "rule_evidence": rule_evidence,
        "rule_evidence_basis": "training partition, dataset failure_risk label",
        "samples": samples,
    }
