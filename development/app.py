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
    vectorizer, model = pc.load_artifacts()
except Exception as exc:  # noqa: BLE001
    vectorizer = model = None
    app.logger.error("Failed to load artifacts: %s", exc)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        if vectorizer is None or model is None:
            raise RuntimeError("Model artifacts not loaded.")

        rgb = pc.predict_rgb(text, vectorizer, model)
        return jsonify({"input": text, "rgb": rgb, "hex": pc.to_hex(rgb)})
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True)
