"""
Evaluate the saved model on a real-world phone photo validation dataset.

Expected JSON format:
[
  {
    "image": "<base64>",
    "concentration": 0.44,
    "filename": "IMG_1234.HEIC"
  }
]
"""

import json
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.extract_features import process_dataset_records


ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_artifacts")
DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phone_validation_dataset.json")


def load_artifacts(artifacts_dir=ARTIFACTS_DIR):
    model = joblib.load(os.path.join(artifacts_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))

    with open(os.path.join(artifacts_dir, "feature_columns.json"), "r") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


def evaluate_dataset(dataset_path=DEFAULT_DATASET):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Validation dataset not found at {dataset_path}. "
            "Add a phone_validation_dataset.json file to run real-world validation."
        )

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    normalized_records = []
    for index, item in enumerate(dataset):
        normalized_records.append({
            "image": item["image"],
            "concentration": float(item["concentration"]),
            "filename": item.get("filename", f"validation_{index}.jpg"),
            "source_id": item.get("source_id", item.get("filename", f"validation_{index}.jpg")),
            "is_augmented": False,
        })

    df = process_dataset_records(normalized_records)
    feature_columns = None
    model, scaler, feature_columns = load_artifacts()

    X = df.reindex(columns=feature_columns, fill_value=0).values
    y = df["concentration"].values

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    metrics = {
        "samples": int(len(y)),
        "r2": float(r2_score(y, predictions)),
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y, predictions)),
    }

    print("\nPHONE VALIDATION RESULTS")
    print("=" * 40)
    print(f"Samples: {metrics['samples']}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f} mg/L")
    print(f"MAE: {metrics['mae']:.4f} mg/L")

    return metrics


if __name__ == "__main__":
    evaluate_dataset()
