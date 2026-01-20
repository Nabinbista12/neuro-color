"""Minimal Flask server to expose the color predictor.
Run with: python app.py
Then open http://127.0.0.1:5000/ in your browser.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import predict_color as pc

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

try:
    vectorizer, models = pc.load_artifacts()
except Exception as exc:  # noqa: BLE001
    vectorizer, models = None, {}
    app.logger.error("Failed to load artifacts: %s", exc)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    # Default to svm; if not available, fall back to first loaded model.
    requested = str(data.get("model", "svm")).strip().lower()
    
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    
    allowed_models = set(models.keys())
    if not allowed_models:
        return jsonify({"error": "No models loaded. Add svm.joblib, ridge.joblib, or randomforest.joblib."}), 500

    model_type = requested if requested in allowed_models else next(iter(allowed_models))

    try:
        if vectorizer is None or not models:
            raise RuntimeError("Model artifacts not loaded.")

        rgb = pc.predict_rgb(text, vectorizer, models, model_type)
        return jsonify({"input": text, "rgb": rgb, "hex": pc.to_hex(rgb), "model": model_type})
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True)
