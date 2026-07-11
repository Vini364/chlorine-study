# Project Structure Documentation

## Overview
This document describes the reorganized project structure for the Chlorine Vision Study.

## Directory Layout

```
chlorine-study/
├── README.md                          # Main project README
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules (updated)
├── .env.example                       # Environment variables template
├── STRUCTURE.md                       # This file
│
├── src/                               # Source code (new)
│   ├── __init__.py
│   ├── main.py                        # FastAPI entry point
│   ├── features.py                    # Feature extraction (was extract_features.py)
│   ├── cnn.py                         # CNN model logic (was CNN.py)
│   ├── preprocessing.py               # Data preprocessing (was augment_images.py)
│   ├── utils.py                       # Utilities (was masks.py)
│   ├── training.py                    # Model training (was train_models.py)
│   ├── training_mobile.py             # Mobile model training (was train_mobile_deployable_model.py)
│   ├── evaluation.py                  # Model evaluation (was evaluate_phone_validation.py)
│   └── pipeline.py                    # Full training pipeline (was run_full_pipeline.py)
│
├── scripts/                           # One-off utility scripts (new)
│   ├── __init__.py
│   ├── convert_images_to_json.py      # Convert images to JSON format
│   ├── predict_single_image.py        # Predict on single image
│   ├── export_poster_metrics.py       # Export metrics for posters
│   ├── test_resolution.py             # Test image resolution
│   ├── check_overfitting.py           # Check for overfitting
│   ├── compare_dataset.py             # Compare datasets
│   └── beer_lambert.py                # Beer Lambert law calculations
│
├── data/                              # Data directory (new - gitignored)
│   ├── raw/                           # Original datasets
│   │   ├── Dataset_1/
│   │   ├── Dataset_2/
│   │   └── Dataset_3/
│   ├── processed/                     # Processed/generated datasets
│   │   ├── phone_holdout_dataset.json
│   │   ├── augmented_image_data.json
│   │   ├── new_dataset.json
│   │   ├── new_dataset_2.json
│   │   └── new_dataset_3.json
│   └── features/                      # Extracted features
│       ├── phone_holdout_features.csv
│       └── features_data.csv
│
├── models/                            # Models & artifacts (new - gitignored)
│   ├── saved_artifacts/               # Trained model artifacts
│   │   ├── scaler.pkl
│   │   ├── best_model.pkl
│   │   ├── feature_columns.json
│   │   ├── model_metadata.json
│   │   └── candidate_results.json
│   ├── best.pt                        # YOLO model weights
│   └── checkpoints/                   # Training checkpoints
│
├── notebooks/                         # Jupyter notebooks (gitignored)
│   └── (*.ipynb files)
│
├── dataset/                           # Original raw data (reference)
│   ├── Dataset_1/
│   ├── Dataset_2/
│   └── Dataset_3/
│
└── chlorine-vision-study/             # Mobile app (separate project)
    ├── mobile/
    ├── README.md
    └── ...
```

## Key Changes from Original Structure

### What Moved Where

| Old Path | New Path | Purpose |
|----------|----------|---------|
| `chlorine-vision-study/main.py` | `src/main.py` | FastAPI server entry point |
| `chlorine-vision-study/training/extract_features.py` | `src/features.py` | Feature extraction module |
| `chlorine-vision-study/training/CNN.py` | `src/cnn.py` | CNN model implementation |
| `chlorine-vision-study/training/augment_images.py` | `src/preprocessing.py` | Data augmentation |
| `chlorine-vision-study/training/train_models.py` | `src/training.py` | Model training logic |
| `chlorine-vision-study/training/train_mobile_deployable_model.py` | `src/training_mobile.py` | Mobile model training |
| `chlorine-vision-study/training/evaluate_phone_validation.py` | `src/evaluation.py` | Model evaluation |
| `chlorine-vision-study/training/run_full_pipeline.py` | `src/pipeline.py` | Full training pipeline |
| Utility scripts (7 files) | `scripts/` | One-off utility scripts |
| `chlorine-vision-study/training/phone_holdout_*.json` | `data/processed/` | Processed datasets |
| `chlorine-vision-study/training/phone_holdout_*.csv` | `data/features/` | Extracted features |
| `chlorine-vision-study/training/saved_artifacts/` | `models/saved_artifacts/` | Model artifacts |
| `chlorine-vision-study/training/best.pt` | `models/best.pt` | YOLO model weights |

### Updated Imports

All Python files have been updated to use the new structure:

- `from training.extract_features import ...` → `from src.features import ...`
- Hardcoded paths now use dynamic `PROJECT_ROOT` calculation
- YOLO model loads from `models/best.pt` instead of hardcoded path
- Artifacts load from `models/saved_artifacts/` instead of `training/saved_artifacts/`

## Benefits of New Structure

1. **Clarity**: Clear separation between source code, scripts, data, and models
2. **Maintainability**: Easier to find and modify specific components
3. **Scalability**: Better organized for future growth
4. **Python best practices**: Follows standard package structure
5. **Version control**: Data and models are gitignored, only code is tracked
6. **Portability**: Paths are computed relative to project root, not absolute

## Usage

### Running the FastAPI Server
```bash
cd /Users/vineshuniyal/Desktop/chlorine-study
python -m src.main
```

### Running a Training Pipeline
```bash
python -m src.pipeline
```

### Using Evaluation
```bash
python -m src.evaluation
```

### Running Utility Scripts
```bash
python scripts/predict_single_image.py path/to/image.jpg
python scripts/convert_images_to_json.py
```

## Notes

- All large data files (models, datasets, CSVs) are gitignored
- `.gitignore` has been updated to track the new paths
- Old paths are still listed in `.gitignore` for backward compatibility
- The `chlorine-vision-study/` directory can still be used for the mobile app
- Original `chlorine-vision-study/training/` directory is now mostly empty (files moved)

## Future Improvements

- Consider adding `tests/` directory for unit tests
- Add `docs/` directory for documentation
- Consider adding CI/CD configuration files (GitHub Actions workflows)
- Add type checking configuration (mypy, pyright)
