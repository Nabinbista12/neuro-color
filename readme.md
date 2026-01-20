# NeuroColor

## Artificial Intelligence Project

## Milestone
- Milestone 1 at January 7. 

## Development setup
- Place model artifacts in `main/code/development/`:
	- `vectorizer.joblib` (required)
	- Optional models: `svm.joblib`, `ridge.joblib`, `randomforest.joblib`
- The app will pick up whichever models exist. If a selected model isn't present, it will fall back to the first available.
- Large `.joblib` files are ignored by git to prevent push errors.