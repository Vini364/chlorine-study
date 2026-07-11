"""
Evaluate the saved model on a real-world phone photo validation dataset.

This module provides utilities to validate trained models against real-world phone photo data.

Expected JSON format:
    [
        {
            "image": "<base64>",
            "concentration": 0.44,
            "filename": "IMG_1234.HEIC"
        }
    ]
"""

import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import process_dataset_records

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_artifacts")
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "processed", "phone_holdout_dataset.json")
FEATURES_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "features", "phone_holdout_features.csv")

# Concentration range bins for analysis
CONCENTRATION_BINS = [0, 0.5, 1.5, 3.0, float("inf")]
CONCENTRATION_LABELS = ["Low", "Medium", "High", "Very High"]

# Model artifacts to skip during evaluation
SKIP_ARTIFACTS = {"scaler.pkl", "best_model.pkl"}


def load_artifacts(artifacts_dir: str = ARTIFACTS_DIR) -> Tuple[Any, List[str]]:
    """Load model artifacts from disk.
    
    Args:
        artifacts_dir: Path to the directory containing artifacts.
        
    Returns:
        Tuple of (scaler, feature_columns).
        
    Raises:
        FileNotFoundError: If required artifact files are not found.
    """
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    feature_cols_path = os.path.join(artifacts_dir, "feature_columns.json")
    
    scaler = joblib.load(scaler_path)
    
    with open(feature_cols_path, "r") as f:
        feature_columns = json.load(f)
    
    return scaler, feature_columns

def explore_dataset(df: pd.DataFrame) -> None:
    """Generate visualization of features vs concentration.
    
    Args:
        df: DataFrame containing features and concentration values.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != "concentration"]
    
    n_cols = 5
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_columns):
        axes[i].scatter(
            df["concentration"],
            df[feature],
            alpha=0.4,
            s=10
        )
        axes[i].set_xlabel("Actual Concentration (mg/L)", fontsize=8)
        axes[i].set_ylabel(feature, fontsize=8)
        axes[i].set_title(f"{feature} vs Concentration", fontsize=9)
    
    # Hide unused subplots
    for j in range(len(numeric_columns), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()


def _normalize_dataset_records(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize dataset records to a consistent format.
    
    Args:
        dataset: Raw dataset records from JSON.
        
    Returns:
        List of normalized records with required fields.
    """
    normalized = []
    for index, item in enumerate(dataset):
        normalized.append({
            "image": item["image"],
            "concentration": float(item["concentration"]),
            "filename": item.get("filename", f"validation_{index}.jpg"),
            "source_id": item.get("source_id", item.get("filename", f"validation_{index}.jpg")),
            "is_augmented": False,
        })
    return normalized


def _calculate_range_metrics(group_data: pd.DataFrame, group_name: str) -> None:
    """Print evaluation metrics for a concentration range group.
    
    Args:
        group_data: DataFrame subset for a specific concentration range.
        group_name: Name of the concentration range (e.g., "Low", "Medium").
    """
    if len(group_data) < 2:
        return
    
    actual = group_data["actual"]
    predicted = group_data["predicted"]
    abs_error = group_data["abs_error"]
    
    print(f"\n{group_name}")
    print(f"Samples: {len(group_data)}")
    print(f"R²: {r2_score(actual, predicted):.4f}")
    print(f"MAE: {mean_absolute_error(actual, predicted):.4f}")
    print(f"RMSE: {mean_squared_error(actual, predicted) ** 0.5:.4f}")
    print(f"Median Error: {abs_error.median():.4f}")
    print("\nWorst Samples in this range:")
    print(
        group_data[["actual", "predicted", "abs_error"]]
        .sort_values("abs_error", ascending=False)
        .head(10)
    )


def evaluate_dataset(dataset_path: str = DEFAULT_DATASET) -> None:
    """Evaluate model performance on validation dataset.
    
    Args:
        dataset_path: Path to the validation dataset JSON file.
        
    Raises:
        FileNotFoundError: If dataset file is not found.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Validation dataset not found at {dataset_path}. "
            "Add a phone_validation_dataset.json file to run real-world validation."
        )
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    normalized_records = _normalize_dataset_records(dataset)
    df = process_dataset_records(normalized_records, FEATURES_CSV_PATH)
    
    scaler, feature_columns = load_artifacts()
    
    X = df.reindex(columns=feature_columns, fill_value=0).values
    y = df["concentration"].values
    X_scaled = scaler.transform(X)
    
    # Evaluate all models
    model_paths = sorted(glob.glob(os.path.join(ARTIFACTS_DIR, "*.pkl")))
    
    print("\n" + "=" * 50)
    print("      VALIDATION REPORT        ")
    print("=" * 50)
    
    for path in model_paths:
        filename = os.path.basename(path)
        if filename in SKIP_ARTIFACTS:
            continue
        
        model = joblib.load(path)
        predictions = model.predict(X_scaled)
        
        analysis_df = pd.DataFrame({
            "actual": y,
            "predicted": predictions
        })
        analysis_df["abs_error"] = abs(
            analysis_df["actual"] - analysis_df["predicted"]
        )
        analysis_df["range"] = pd.cut(
            analysis_df["actual"],
            bins=CONCENTRATION_BINS,
            labels=CONCENTRATION_LABELS
        )
        
        print(f"\nModel: {filename}")
        print("\nError by Concentration Range")
        
        for group_name, subset in analysis_df.groupby("range"):
            _calculate_range_metrics(subset, group_name)


def _parse_arguments() -> str:
    """Parse command line arguments.
    
    Returns:
        Path to the target dataset file.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]
    return DEFAULT_DATASET


def _get_user_choice() -> str:
    """Prompt user to choose between CSV or JSON input.
    
    Returns:
        User choice ('csv' or 'json').
        
    Raises:
        SystemExit: If invalid choice is provided.
    """
    print("Choose input format:")
    print("  'csv'  - Read from CSV file")
    print("  'json' - Read from JSON file and create CSV")
    choice = input("Enter 'csv' or 'json': ").strip().lower()
    
    if choice not in ("csv", "json"):
        print("Invalid choice. Please enter 'csv' or 'json'.")
        sys.exit(1)
    
    return choice


def main() -> None:
    """Main entry point for the evaluation script."""
    dataset_path = _parse_arguments()
    choice = _get_user_choice()
    
    if choice == "csv":
        if not os.path.exists(FEATURES_CSV_PATH):
            print(f"CSV file not found at {FEATURES_CSV_PATH}.")
            sys.exit(1)
        df = pd.read_csv(FEATURES_CSV_PATH)
        explore_dataset(df)
    elif choice == "json":
        if not os.path.exists(dataset_path):
            print(f"JSON file not found at {dataset_path}.")
            sys.exit(1)
        evaluate_dataset(dataset_path)


if __name__ == "__main__":
    main()
