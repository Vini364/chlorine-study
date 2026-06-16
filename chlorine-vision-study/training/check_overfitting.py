"""
Check train vs holdout gaps for the current saved regression model.

This script uses the saved model artifacts from the last full pipeline run and
reports whether the training-to-holdout performance gap looks mild, moderate,
or severe.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.extract_features import process_dataset_records


TRAINING_DIR = Path("training")
ARTIFACTS_DIR = TRAINING_DIR / "saved_artifacts"


def load_json(path: Path):
    with path.open("r") as file:
        return json.load(file)


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def classify_gap(train_metrics, holdout_metrics):
    r2_gap = train_metrics["r2"] - holdout_metrics["r2"]
    rmse_ratio = holdout_metrics["rmse"] / max(train_metrics["rmse"], 1e-9)
    mae_ratio = holdout_metrics["mae"] / max(train_metrics["mae"], 1e-9)

    if r2_gap > 0.12 or rmse_ratio > 4.5 or mae_ratio > 4.5:
        return "severe"
    if r2_gap > 0.05 or rmse_ratio > 2.5 or mae_ratio > 2.5:
        return "moderate"
    return "mild"


def main():
    feature_columns = load_json(ARTIFACTS_DIR / "feature_columns.json")
    model_metadata = load_json(ARTIFACTS_DIR / "model_metadata.json")

    model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")

    train_df = pd.read_csv(TRAINING_DIR / "features_data.csv")
    train_original_df = train_df[train_df["is_augmented"] == False].copy()

    holdout_records = load_json(TRAINING_DIR / "phone_holdout_dataset.json")
    holdout_df = process_dataset_records(holdout_records)

    def predict(df: pd.DataFrame):
        X = df.reindex(columns=feature_columns, fill_value=0).values
        y = df["concentration"].values.astype(float)
        predictions = model.predict(scaler.transform(X))
        return compute_metrics(y, predictions)

    train_all_metrics = predict(train_df)
    train_original_metrics = predict(train_original_df)
    holdout_metrics = predict(holdout_df)

    gap_r2 = train_original_metrics["r2"] - holdout_metrics["r2"]
    gap_rmse = holdout_metrics["rmse"] / max(train_original_metrics["rmse"], 1e-9)
    gap_mae = holdout_metrics["mae"] / max(train_original_metrics["mae"], 1e-9)
    severity = classify_gap(train_original_metrics, holdout_metrics)

    summary = {
        "model_name": model_metadata["model_name"],
        "counts": {
            "train_rows_with_augmentation": int(len(train_df)),
            "train_original_rows": int(len(train_original_df)),
            "holdout_original_rows": int(len(holdout_df)),
        },
        "train_all_rows": train_all_metrics,
        "train_original_only": train_original_metrics,
        "holdout_original_only": holdout_metrics,
        "cross_validation": {
            "cv_r2_mean": float(model_metadata["cv_r2_mean"]),
            "cv_r2_std": float(model_metadata["cv_r2_std"]),
        },
        "gaps": {
            "train_original_minus_holdout_r2": float(gap_r2),
            "holdout_to_train_original_rmse_ratio": float(gap_rmse),
            "holdout_to_train_original_mae_ratio": float(gap_mae),
            "severity": severity,
        },
    }

    print(json.dumps(summary, indent=2))

    print("\nInterpretation:")
    if severity == "mild":
        print("The model shows only a small train-to-holdout gap and does not appear meaningfully overfit.")
    elif severity == "moderate":
        print("The model shows a noticeable train-to-holdout gap, so there is some overfitting, but holdout performance is still strong.")
    else:
        print("The model shows a large train-to-holdout gap, suggesting substantial overfitting.")


if __name__ == "__main__":
    main()
