"""Feature preparation for the Stage 9 prediction service.

The final saved pipeline already recreates ``Sound_Level_Code`` through
``SoundTertileEncoder``.  This module rebuilds the remaining PE1 engineered
features from the raw sensor values entered by the user.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Iterable

import pandas as pd


RAW_SENSOR_FIELDS = [
    "S1_Temp",
    "S2_Temp",
    "S3_Temp",
    "S4_Temp",
    "S1_Light",
    "S2_Light",
    "S3_Light",
    "S4_Light",
    "S1_Sound",
    "S2_Sound",
    "S3_Sound",
    "S4_Sound",
    "S5_CO2",
    "S5_CO2_Slope",
    "S6_PIR",
    "S7_PIR",
]

REQUIRED_INPUT_FIELDS = ["time", *RAW_SENSOR_FIELDS]


class InputValidationError(ValueError):
    """Raised when a prediction request contains invalid user input."""


def _require_number(payload: dict[str, Any], field: str) -> float:
    if field not in payload or payload[field] in (None, ""):
        raise InputValidationError(f"Missing required field: {field}")

    try:
        value = float(payload[field])
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{field} must be a number") from exc

    if not math.isfinite(value):
        raise InputValidationError(f"{field} must be a finite number")

    return value


def _time_to_minutes(value: Any) -> float:
    if value in (None, ""):
        raise InputValidationError("Missing required field: time")

    text = str(value).strip()
    parsed = None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        raise InputValidationError("time must use HH:MM or HH:MM:SS format")

    return parsed.hour * 60 + parsed.minute + parsed.second / 60


def _co2_level_code(co2: float) -> int:
    """Match the fixed PE1 CO2 bands: <800, 800-1000, >1000."""

    if co2 < 800:
        return 0
    if co2 <= 1000:
        return 1
    return 2


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def build_model_frame(
    payload: dict[str, Any], expected_features: list[str]
) -> pd.DataFrame:
    """Validate raw user input and create one model-ready DataFrame row."""

    if not isinstance(payload, dict):
        raise InputValidationError("Request body must be a JSON object")

    numbers = {field: _require_number(payload, field) for field in RAW_SENSOR_FIELDS}

    for field in ("S1_Light", "S2_Light", "S3_Light", "S4_Light"):
        if numbers[field] < 0:
            raise InputValidationError(f"{field} cannot be negative")

    for field in ("S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound", "S5_CO2"):
        if numbers[field] < 0:
            raise InputValidationError(f"{field} cannot be negative")

    for field in ("S6_PIR", "S7_PIR"):
        if numbers[field] not in (0.0, 1.0):
            raise InputValidationError(f"{field} must be either 0 or 1")
        numbers[field] = int(numbers[field])

    temperatures = [numbers[f"S{i}_Temp"] for i in range(1, 5)]
    lights = [numbers[f"S{i}_Light"] for i in range(1, 5)]
    sounds = [numbers[f"S{i}_Sound"] for i in range(1, 5)]

    engineered = {
        "S5_CO2": numbers["S5_CO2"],
        "S5_CO2_Slope": numbers["S5_CO2_Slope"],
        "S6_PIR": numbers["S6_PIR"],
        "S7_PIR": numbers["S7_PIR"],
        "Time_of_Day_Minutes": _time_to_minutes(payload.get("time")),
        "Avg_Temp": _mean(temperatures),
        "Avg_Light": _mean(lights),
        "Avg_Sound": _mean(sounds),
        "PIR_Activity": numbers["S6_PIR"] + numbers["S7_PIR"],
        "Temp_Spread": max(temperatures) - min(temperatures),
        "Light_Spread": max(lights) - min(lights),
        "CO2_Level_Code": _co2_level_code(numbers["S5_CO2"]),
        "S1_Temp": numbers["S1_Temp"],
        "S2_Temp": numbers["S2_Temp"],
        "S3_Temp": numbers["S3_Temp"],
        "S4_Temp": numbers["S4_Temp"],
        "S1_Light": numbers["S1_Light"],
        "S2_Light": numbers["S2_Light"],
        "S3_Light": numbers["S3_Light"],
        "S4_Light": numbers["S4_Light"],
        "S1_Sound": numbers["S1_Sound"],
        "S2_Sound": numbers["S2_Sound"],
        "S3_Sound": numbers["S3_Sound"],
        "S4_Sound": numbers["S4_Sound"],
    }

    missing_for_model = [name for name in expected_features if name not in engineered]
    extra_for_model = [name for name in engineered if name not in expected_features]
    if missing_for_model or extra_for_model:
        raise RuntimeError(
            "Feature contract does not match the saved model metadata. "
            f"Missing: {missing_for_model}; extra: {extra_for_model}"
        )

    return pd.DataFrame([[engineered[name] for name in expected_features]], columns=expected_features)
