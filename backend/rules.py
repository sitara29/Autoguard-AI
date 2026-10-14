"""Transparent, rule-based maintenance advisories.

These rules are shared by the training script (which measures how strongly each
rule is associated with the dataset's failure_risk label) and by the API (which
applies them to a single vehicle). They are decision-support suggestions drawn
from general maintenance practice; they are NOT outputs of the ML model.

Each `test` accepts either a dict of feature values (one vehicle) or a pandas
DataFrame (many vehicles) and returns a bool / boolean Series.
"""
from __future__ import annotations

from typing import Any

RULES: list[dict[str, Any]] = [
    {
        "id": "engine_overheat",
        "component": "engine",
        "system": "engine_cooling",
        "severity": "high",
        "condition": "engine_temp_c >= 110",
        "test": lambda v: v["engine_temp_c"] >= 110,
        "message": "Engine temperature is elevated in this input. Check the cooling system and follow the vehicle manufacturer's guidance.",
    },
    {
        "id": "oil_quality_low",
        "component": "lubrication",
        "system": "lubrication",
        "severity": "medium",
        "condition": "oil_quality_pct < 60",
        "test": lambda v: v["oil_quality_pct"] < 60,
        "message": "The oil-quality index is low. Check oil condition and the scheduled oil-service interval.",
    },
    {
        "id": "brake_pad_low",
        "component": "brakes",
        "system": "brakes",
        "severity": "high",
        "condition": "brake_pad_mm <= 3.0",
        "test": lambda v: v["brake_pad_mm"] <= 3.0,
        "message": "Reported brake-pad thickness is low. Have the braking system inspected by a qualified technician.",
    },
    {
        "id": "coolant_low",
        "component": "cooling",
        "system": "cooling",
        "severity": "medium",
        "condition": "coolant_level_pct < 60",
        "test": lambda v: v["coolant_level_pct"] < 60,
        "message": "Reported coolant level is low. Check the correct fill level and inspect for possible leaks.",
    },
    {
        "id": "battery_voltage_low",
        "component": "battery",
        "system": "battery",
        "severity": "medium",
        "condition": "battery_voltage < 12.0",
        "test": lambda v: v["battery_voltage"] < 12.0,
        "message": "Battery voltage is on the low side of the dataset's observed range. Consider a battery and charging-system test.",
    },
    {
        "id": "age_or_mileage_high",
        "component": "general_wear",
        "system": "general_maintenance",
        "severity": "medium",
        "condition": "vehicle_age_years >= 10 or mileage_km >= 150000",
        "test": lambda v: (v["vehicle_age_years"] >= 10) | (v["mileage_km"] >= 150_000),
        "message": "Age or mileage is relatively high. Consider a comprehensive inspection in line with the manufacturer's service schedule.",
    },
]

SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


def triggered_rules(values: dict[str, float]) -> list[dict[str, Any]]:
    """Rules that fire for one vehicle's feature values."""
    return [rule for rule in RULES if bool(rule["test"](values))]
