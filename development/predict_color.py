
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "vectorizer.joblib"
# Expected model artifact filenames the user will drop in the development folder.
# Support preferred names and a couple of legacy synonyms for convenience.
MODEL_PATHS: Dict[str, Path] = {
    "svm": BASE_DIR / "svm.joblib",
    "ridge": BASE_DIR / "ridge.joblib",
    "random_forest": BASE_DIR / "random_forest.joblib",
}


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s", " ", text)
    return text.strip()


def remove_numbers(text: str) -> str:
    text = str(text)
    text = re.sub(r"\b\d+[a-zA-Z]*\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    return remove_numbers(clean_text(text))


def load_artifacts():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Missing vectorizer file: {VECTORIZER_PATH.name}. Place it in {BASE_DIR}."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)

    models: Dict[str, object] = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            models[name] = joblib.load(path)
    return vectorizer, models


def predict_rgb(text: str, vectorizer, models: Dict[str, object], model_type: str = "svm") -> Tuple[int, int, int]:
    prepped = preprocess(text)
    features = vectorizer.transform([prepped])

    model_key = (model_type or "svm").lower()
    if model_key not in models:
        raise ValueError(f"Unsupported model '{model_key}'")

    preds = models[model_key].predict(features)

    rgb = np.clip(np.rint(np.asarray(preds).reshape(-1)[:3]), 0, 255).astype(int)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    parser = argparse.ArgumentParser(description="Predict an RGB color from a Moodx ")
    parser.add_argument("text", nargs=argparse.REMAINDER, help="Words describing the color.")
    args = parser.parse_args()

    user_text = " ".join(args.text).strip()

    # If nothing was provided on the CLI, ask interactively instead of erroring.
    if not user_text:
        user_text = input("Enter how you feel right now: ").strip()

    if not user_text:
        parser.error("Please provide a text prompt, e.g. python predict_color.py \"calm ocean\"")

    vectorizer, models = load_artifacts()
    # Prefer SVM if available; otherwise use the first available model.
    default_model = "svm" if "svm" in models else (next(iter(models)) if models else None)
    if not default_model:
        raise RuntimeError("No model artifacts found. Add svm.joblib, ridge.joblib, or randomforest.joblib to the development folder.")
    rgb = predict_rgb(user_text, vectorizer, models, default_model)
    hex_code = to_hex(rgb)

    result = {"input": user_text, "rgb": rgb, "hex": hex_code}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
