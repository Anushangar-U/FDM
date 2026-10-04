"""Stage 9 backend and Stage 10 frontend entry point.

Run from the repository root with:

    python system/app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

from pathlib import Path
import json
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

app = Flask(__name__, template_folder="templates", static_folder="static")


def _load_metadata() -> dict:
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    RANDOM_SOURCE_ERROR = None
except Exception as exc:
    RANDOM_SOURCE = None
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


@app.get("/api/random-input")
def random_input():
    """Return one random real sensor record without exposing its target label."""

    if RANDOM_SOURCE is None or RANDOM_SOURCE.empty:
        return (
            jsonify(
                {
                    "error": "Random sample data could not be loaded.",
                    "details": RANDOM_SOURCE_ERROR,
                }
            ),
            503,
        )

    row = RANDOM_SOURCE.iloc[secrets.randbelow(len(RANDOM_SOURCE))]
    sample = {"time": str(row["Time"])}

    for field in RAW_RANDOM_FIELDS:
        if field == "Time":
            continue
        value = row[field]
        if field in ("S6_PIR", "S7_PIR"):
            sample[field] = int(value)
        else:
            sample[field] = float(value)

    return jsonify({"sample": sample, "available_records": int(len(RANDOM_SOURCE))})


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
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
