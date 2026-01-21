# NeuroColor

## Artificial Intelligence Project

## Overview
NeuroColor converts a short text description of a mood, object, or vibe into a representative RGB color. It uses classical ML models trained on text–color pairs. You can run it as a web app (Flask) or via a small CLI script.

## How It Works
- Preprocess: Lowercase, clean punctuation, and remove numbers.
- Vectorize: Transform text with `vectorizer.joblib` (required).
- Predict: Pass features to a selected model (`svm`, `ridge`, `grid`, or `random_forest`) to output three values.
- Postprocess: Clamp to [0,255], format as RGB and HEX.
- Serve: `app.py` exposes a POST `/predict` endpoint and a simple UI in `index.html`.

## Milestone
- Milestone 1 at January 7. 

## Development setup
	- `vectorizer.joblib` (required)
	- Optional models: `svm.joblib`, `ridge.joblib`, `grid.joblib`, `random_forest.joblib`
	- The app will pick up whichever models exist. If a selected model isn't present, it will error gracefully.

## Models

### Available Models
- **SVM** (`svm.joblib`): Support Vector Machine - a linear/non-linear classifier
- **Ridge** (`ridge.joblib`): Ridge Regression - linear regression with L2 regularization
- **Grid** (`grid.joblib`): GridSearchCV optimized model - tuned hyperparameters
- **Random Forest** (`random_forest.joblib`): Ensemble method using multiple decision trees

## Generating Model Artifacts

### Random Forest Model
To generate the `random_forest.joblib` file:
1. Open and run the notebook: `main/code/alogrithm/random_forest.ipynb`
2. This will train the random forest model and export it to `random_forest.joblib`
3. Move the generated `.joblib` file to `main/code/development/` folder

### Other Models
Similar notebooks exist for other models:
- `main/code/alogrithm/svm.ipynb` - generates `svm.joblib`
- `main/code/alogrithm/ridge.ipynb` - generates `ridge.joblib`

## Running the Application
```bash
# Development server
python main/code/development/app.py

# CLI prediction
python main/code/development/predict_color.py "calm ocean" --model svm
```