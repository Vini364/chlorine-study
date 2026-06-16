# Handoff Guide

This repository has two main parts:

- `training/`: model training, feature extraction, evaluation, and saved artifacts
- `mobile/`: Expo React Native app that runs chlorine prediction on-device

The project goal is to estimate chlorine concentration from a phone image of a
pink DPD-treated water sample and return the result in `mg/L`.

## Current Source Of Truth

- Full Python pipeline model metadata: `training/saved_artifacts/model_metadata.json`
- Mobile app model metadata: `training/saved_artifacts/mobile_model_metadata.json`
- Candidate model comparison: `training/saved_artifacts/candidate_results.json`
- Poster metrics summary: `training/saved_artifacts/poster_metrics.json`

## Repo Map

### Top level

- `README.md`: main project overview, run instructions, and high-level workflow.
- `HANDOFF.md`: file guide.
- `requirements.txt`: Python dependencies for training and the optional FastAPI API.
- `main.py`: optional FastAPI backend for Python-side prediction and debugging.
- `convert_images_to_json.py`: converts folder-based raw sample images into the JSON dataset used by training.
- `.gitignore`: ignores caches, local runtime artifacts, and regenerated training intermediates.
- `.env.example`: placeholder environment configuration example.
- `.dockerignore`: leftover ignore file for container contexts; not central to the current workflow.

### `training/`

- `training/new_dataset.json`: main labeled dataset used for retraining. Images are stored as base64 strings alongside concentration labels.
- `training/run_full_pipeline.py`: main end-to-end ML pipeline. Splits originals before augmentation, extracts features, trains candidate models, ranks them with overfitting-aware logic, and saves artifacts.
- `training/augment_images.py`: image augmentation code for generating more training examples from the original phone photos.
- `training/extract_features.py`: decodes images, crops to the test-tube / pink-liquid region, and extracts RGB/HSV/LAB/color/texture features.
- `training/train_mobile_deployable_model.py`: trains and exports the smaller mobile-safe model that can be reproduced inside the Expo app.
- `training/predict_single_image.py`: helper script for running the saved Python model on one image file.
- `training/check_overfitting.py`: compares train and holdout performance for the saved full pipeline model and classifies the gap as mild, moderate, or severe.
- `training/export_poster_metrics.py`: writes a compact summary JSON for poster and presentation use.
- `training/evaluate_phone_validation.py`: optional evaluation script for a separate external phone-photo validation dataset if one is collected later.
- `training/train_models.py`: older exploratory training script kept for history. The main workflow is `training/run_full_pipeline.py`.

### `training/saved_artifacts/`

- `training/saved_artifacts/best_model.pkl`: saved full Python regression model selected by the main pipeline.
- `training/saved_artifacts/scaler.pkl`: saved scaler for the Python-side feature vector.
- `training/saved_artifacts/feature_columns.json`: exact feature order expected by the saved full model.
- `training/saved_artifacts/model_metadata.json`: selected full pipeline model name, metrics, and overfitting summary.
- `training/saved_artifacts/candidate_results.json`: results for all candidate full-pipeline models.
- `training/saved_artifacts/mobile_model.json`: exported deployable model payload for mobile/on-device inference.
- `training/saved_artifacts/mobile_model_metadata.json`: metrics and metadata for the mobile model.
- `training/saved_artifacts/poster_metrics.json`: compact metrics bundle used for slides, posters, and quick summaries.

### `mobile/`

- `mobile/App.tsx`: main app flow and screen logic for home, capture, camera, processing, results, and scan history.
- `mobile/index.ts`: Expo app entry point.
- `mobile/app.json`: Expo configuration and permissions.
- `mobile/package.json`: app dependencies and scripts.
- `mobile/package-lock.json`: locked npm dependency tree for reproducible installs.
- `mobile/tsconfig.json`: TypeScript config.
- `mobile/README.md`: short app-specific run notes.

### `mobile/src/`

- `mobile/src/analyzer.ts`: on-device inference logic. Recomputes deployable features in JavaScript and applies the exported mobile model.
- `mobile/src/mobile_model.json`: actual model file bundled into the app.
- `mobile/src/scanHistory.ts`: local storage and location handling for saved scans.
- `mobile/src/styles.ts`: app styling definitions.

### `mobile/assets/`

- `mobile/assets/icon.png`: primary app icon.
- `mobile/assets/splash-icon.png`: splash screen image.
- `mobile/assets/favicon.png`: favicon for Expo/web contexts.
- `mobile/assets/android-icon-background.png`: Android adaptive icon background asset.
- `mobile/assets/android-icon-foreground.png`: Android adaptive icon foreground asset.
- `mobile/assets/android-icon-monochrome.png`: Android monochrome icon asset.

### `poster_assets/`

- `poster_assets/model_comparison_chart.py`: poster chart generator for the overall model comparison figure.
- `poster_assets/model_comparison.png`: generated chart image used for the poster/presentation.

## Recommended Starting Points

- Read `README.md` for the project overview and run commands.
- Read `training/run_full_pipeline.py` for the main ML workflow.
- Read `mobile/App.tsx` and `mobile/src/analyzer.ts` for the app flow and on-device inference.

## Data And Model Flow

1. Raw phone photos are organized by known chlorine concentration.
2. `convert_images_to_json.py` converts those folders into `training/new_dataset.json`.
3. `training/run_full_pipeline.py` loads that dataset and splits original images into training and holdout sets before augmentation.
4. `training/augment_images.py` creates synthetic variations only for the training split.
5. `training/extract_features.py` crops to the chemically relevant region and computes image features.
6. The main pipeline trains multiple regression models and saves the selected Python-side artifacts.
7. `training/train_mobile_deployable_model.py` exports a mobile-safe model using deployable features only.
8. `mobile/src/analyzer.ts` loads that exported mobile model and runs predictions directly on the phone.

## Current Trained Models

- Mobile app model: `Mobile Lasso Regression`
  - metadata: `training/saved_artifacts/mobile_model_metadata.json`
  - role: deployable model used by the Expo app for on-device inference

The tree models achieved stronger raw accuracy in some runs but the currently selected saved models were chosen for better generalization and lower overfitting.

## Common Tasks

Run the mobile app:

```bash
cd mobile
npm install
npx expo start -c
```

Run the optional Python API:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Retrain the full pipeline and refresh exported artifacts:

```bash
python -m training.run_full_pipeline
python -m training.train_mobile_deployable_model
python -m training.export_poster_metrics
```

Check overfitting:

```bash
python -m training.check_overfitting
```

Generate a poster metrics summary:

```bash
python -m training.export_poster_metrics
```

## Notes For Future Work

- The mobile app does not need the Python API for normal inference.
- The Python API is mainly useful for debugging and single-image testing.
- `training/train_models.py` is a exploratory training script. The main pipeline is `training/run_full_pipeline.py`.
- Poster assets belong in `poster_assets/`. 
- The files `training/augmented_image_data.json`, `training/features_data.csv`, and `training/phone_holdout_dataset.json` are generated outputs, not long-term source files.