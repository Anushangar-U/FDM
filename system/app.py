"""Stage 9 backend and Stage 10 frontend entry point.

Run from the repository root with:

    python system/app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

from flask import Flask, jsonify, render_template, request
import joblib


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
