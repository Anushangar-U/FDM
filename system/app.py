"""Stage 9 backend and Stage 10 frontend entry point.

Run from the repository root with:

    python system/app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import random
import secrets
import sys

from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import before loading joblib so the custom transformer class is available.
from src.pe2_transformers import SoundTertileEncoder  # noqa: E402,F401
from system.feature_engineering import (  # noqa: E402
    InputValidationError,
    build_model_frame,
)


MODEL_PATH = ROOT / "models" / "room_occupancy_final_model.joblib"
METADATA_PATH = ROOT / "models" / "room_occupancy_final_model_metadata.json"
RAW_DATA_PATH = ROOT / "data" / "raw" / "Occupancy_Estimation.csv"

RAW_RANDOM_FIELDS = [
    "Time",
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

CONTINUOUS_RANDOM_FIELDS = [
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
]

FIELD_PRECISION = {
    "S1_Temp": 2,
    "S2_Temp": 2,
    "S3_Temp": 2,
    "S4_Temp": 2,
    "S1_Light": 0,
    "S2_Light": 0,
    "S3_Light": 0,
    "S4_Light": 0,
    "S1_Sound": 3,
    "S2_Sound": 3,
    "S3_Sound": 3,
    "S4_Sound": 3,
    "S5_CO2": 0,
    "S5_CO2_Slope": 3,
}

RNG = random.SystemRandom()

app = Flask(__name__, template_folder="templates", static_folder="static")


def _load_metadata() -> dict:
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_input_profiles(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build dataset-informed soft ranges and jitter sizes for each sensor."""

    profiles: dict[str, dict[str, float]] = {}

    for field in CONTINUOUS_RANDOM_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce").dropna()
        if values.empty:
            continue

        quantiles = values.quantile([0.01, 0.25, 0.75, 0.99])
        observed_low = float(values.min())
        observed_high = float(values.max())
        typical_low = float(quantiles.loc[0.01])
        typical_high = float(quantiles.loc[0.99])

        if not math.isfinite(typical_low) or not math.isfinite(typical_high):
            continue

        if typical_low == typical_high:
            typical_low = observed_low
            typical_high = observed_high

        iqr = float(quantiles.loc[0.75] - quantiles.loc[0.25])
        typical_span = max(typical_high - typical_low, 0.0)
        jitter = max(abs(iqr) * 0.10, typical_span * 0.02, 0.001)

        profiles[field] = {
            "typical_low": typical_low,
            "typical_high": typical_high,
            "observed_low": observed_low,
            "observed_high": observed_high,
            "jitter": jitter,
        }

    return profiles


def _jitter_time(value: object) -> str:
    """Move an observed time by up to 20 minutes and wrap within 24 hours."""

    text = str(value).strip()
    try:
        hour, minute, second = (int(part) for part in text.split(":"))
    except (TypeError, ValueError):
        hour, minute, second = 12, 0, 0

    total_seconds = hour * 3600 + minute * 60 + second
    total_seconds = (total_seconds + RNG.randint(-1200, 1200)) % 86400

    new_hour, remainder = divmod(total_seconds, 3600)
    new_minute, new_second = divmod(remainder, 60)
    return f"{new_hour:02d}:{new_minute:02d}:{new_second:02d}"


def _generate_random_input() -> dict[str, float | int | str]:
    """Create a new sensor combination near a real observation, within soft bounds."""

    anchor = RANDOM_SOURCE.iloc[secrets.randbelow(len(RANDOM_SOURCE))]
    sample: dict[str, float | int | str] = {"time": _jitter_time(anchor["Time"])}

    for field in CONTINUOUS_RANDOM_FIELDS:
        profile = INPUT_PROFILES[field]
        anchor_value = float(anchor[field])
        generated = anchor_value + RNG.gauss(0.0, profile["jitter"])
        generated = min(
            max(generated, profile["typical_low"]),
            profile["typical_high"],
        )
        sample[field] = round(generated, FIELD_PRECISION[field])

    # PIR is binary. Usually preserve the anchor state so generated combinations
    # remain coherent, with a small chance of changing each motion sensor.
    for field in ("S6_PIR", "S7_PIR"):
        value = int(anchor[field])
        if RNG.random() < 0.10:
            value = 1 - value
        sample[field] = value

    return sample


def _sample_real_input() -> dict[str, float | int | str]:
    """Return one unchanged real sensor observation without its target label."""

    row = RANDOM_SOURCE.iloc[secrets.randbelow(len(RANDOM_SOURCE))]
    sample: dict[str, float | int | str] = {"time": str(row["Time"])}

    for field in RAW_RANDOM_FIELDS:
        if field == "Time":
            continue
        value = row[field]
        if field in ("S6_PIR", "S7_PIR"):
            sample[field] = int(value)
        else:
            sample[field] = float(value)

    return sample


def _range_warnings(payload: dict) -> list[dict[str, float | str]]:
    """Return soft out-of-distribution warnings without blocking prediction."""

    warnings: list[dict[str, float | str]] = []

    for field, profile in INPUT_PROFILES.items():
        if field not in payload:
            continue

        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            continue

        if not math.isfinite(value):
            continue

        low = profile["typical_low"]
        high = profile["typical_high"]
        if value < low or value > high:
            warnings.append(
                {
                    "field": field,
                    "value": value,
                    "typical_low": low,
                    "typical_high": high,
                    "message": (
                        f"{field} is unusual for this dataset. "
                        f"Typical range is {low:.3f} to {high:.3f}."
                    ),
                }
            )

    return warnings


METADATA = _load_metadata()
EXPECTED_FEATURES = METADATA["input_features"]
MODEL_NAME = METADATA["selected_model"]

try:
    MODEL = joblib.load(MODEL_PATH)
    MODEL_LOAD_ERROR = None
except Exception as exc:  # keep the UI available with a useful error message
    MODEL = None
    MODEL_LOAD_ERROR = str(exc)

try:
    RANDOM_SOURCE = pd.read_csv(RAW_DATA_PATH, usecols=RAW_RANDOM_FIELDS)
    INPUT_PROFILES = _build_input_profiles(RANDOM_SOURCE)
    RANDOM_SOURCE_ERROR = None
except Exception as exc:
    RANDOM_SOURCE = None
    INPUT_PROFILES = {}
    RANDOM_SOURCE_ERROR = str(exc)


PREDICTION_LABELS = {
    0: "Room predicted to be empty",
    1: "1 person predicted in the room",
    2: "2 people predicted in the room",
    3: "3 people predicted in the room",
}


@app.get("/")
def home():
    return render_template("index.html", model_name=MODEL_NAME)


@app.get("/api/health")
def health():
    if MODEL is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "model_loaded": False,
                    "message": MODEL_LOAD_ERROR,
                }
            ),
            503,
        )

    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "model": MODEL_NAME,
            "target_classes": METADATA["target_classes"],
        }
    )


@app.get("/api/input-ranges")
def input_ranges():
    if not INPUT_PROFILES:
        return jsonify({"error": "Input ranges could not be loaded."}), 503

    public_ranges = {
        field: {
            "typical_low": profile["typical_low"],
            "typical_high": profile["typical_high"],
        }
        for field, profile in INPUT_PROFILES.items()
    }
    return jsonify(
        {
            "ranges": public_ranges,
            "description": "Typical ranges use the 1st to 99th percentiles of the project dataset.",
        }
    )


@app.get("/api/random-input")
def random_input():
    """Generate new bounded sensor values rather than copying a dataset row."""

    if RANDOM_SOURCE is None or RANDOM_SOURCE.empty or not INPUT_PROFILES:
        return (
            jsonify(
                {
                    "error": "Random input data could not be loaded.",
                    "details": RANDOM_SOURCE_ERROR,
                }
            ),
            503,
        )

    sample = _generate_random_input()
    return jsonify(
        {
            "sample": sample,
            "generation": "dataset-informed random values",
        }
    )


@app.get("/api/real-input")
def real_input():
    """Return one real observation sampled from the full raw dataset."""

    if RANDOM_SOURCE is None or RANDOM_SOURCE.empty:
        return (
            jsonify(
                {
                    "error": "Real sample data could not be loaded.",
                    "details": RANDOM_SOURCE_ERROR,
                }
            ),
            503,
        )

    return jsonify(
        {
            "sample": _sample_real_input(),
            "available_records": int(len(RANDOM_SOURCE)),
            "generation": "real dataset record",
        }
    )


@app.post("/api/predict")
def predict():
    if MODEL is None:
        return (
            jsonify(
                {
                    "error": "The trained model could not be loaded.",
                    "details": MODEL_LOAD_ERROR,
                }
            ),
            503,
        )

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Send the input as JSON."}), 400

    try:
        model_input = build_model_frame(payload, EXPECTED_FEATURES)
        warnings = _range_warnings(payload)
        prediction = int(MODEL.predict(model_input)[0])
        probabilities = MODEL.predict_proba(model_input)[0]
        model_classes = getattr(MODEL, "classes_", METADATA["target_classes"])
        classes = [int(value) for value in model_classes]
    except InputValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500

    if prediction not in PREDICTION_LABELS:
        return jsonify({"error": f"Model returned an unexpected class: {prediction}"}), 500

    probability_map = {
        str(class_id): round(float(probability), 6)
        for class_id, probability in zip(classes, probabilities)
    }

    return jsonify(
        {
            "prediction": prediction,
            "message": PREDICTION_LABELS[prediction],
            "probabilities": probability_map,
            "warnings": warnings,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
