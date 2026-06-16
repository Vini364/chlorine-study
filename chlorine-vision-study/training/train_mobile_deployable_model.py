"""
Train a mobile-deployable chlorine model and export it as JSON for the Expo app.

This version intentionally avoids Python-only feature requirements such as Haralick
textures and instead uses color statistics that we can reproduce in JavaScript
from a framed JPEG inside the mobile app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


TRAINING_DIR = Path("training")
FEATURES_CSV = TRAINING_DIR / "features_data.csv"
ARTIFACTS_DIR = TRAINING_DIR / "saved_artifacts"
MOBILE_MODEL_PATH = ARTIFACTS_DIR / "mobile_model.json"
MOBILE_METADATA_PATH = ARTIFACTS_DIR / "mobile_model_metadata.json"
MOBILE_APP_MODEL_PATH = Path("mobile/src/mobile_model.json")

DEPLOYABLE_FEATURE_COLUMNS = [
    "mean_R",
    "mean_G",
    "mean_B",
    "std_R",
    "std_G",
    "std_B",
    "norm_r",
    "norm_g",
    "norm_b",
    "ratio_R_G",
    "ratio_R_B",
    "ratio_G_B",
    "diff_R_G",
    "diff_R_B",
    "diff_G_B",
    "mean_H",
    "mean_S",
    "mean_V",
    "std_H",
    "std_S",
    "std_V",
    "mean_L",
    "mean_A",
    "mean_B_lab",
    "std_L",
    "std_A",
    "std_B_lab",
    "brightness",
    "color_saturation",
    "dominant_channel",
]

def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing feature data at {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    missing = [column for column in DEPLOYABLE_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing deployable feature columns: {missing}")

    groups = df["source_id"].astype(str).values if "source_id" in df.columns else df.index.astype(str).values
    y = df["concentration"].values.astype(float)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, y, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[DEPLOYABLE_FEATURE_COLUMNS].values
    X_test = test_df[DEPLOYABLE_FEATURE_COLUMNS].values
    y_train = train_df["concentration"].values.astype(float)
    y_test = test_df["concentration"].values.astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Lasso(alpha=0.0005, max_iter=20000)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    metrics = compute_metrics(y_test, predictions)
    train_metrics = compute_metrics(y_train, model.predict(X_train_scaled))

    export_payload = {
        "model_name": "Mobile Lasso Regression",
        "model_type": "linear",
        "feature_columns": DEPLOYABLE_FEATURE_COLUMNS,
        "unit": "mg/L",
        "coefficients": [float(value) for value in model.coef_.tolist()],
        "intercept": float(model.intercept_),
        "scaler": {
            "mean": [float(value) for value in scaler.mean_.tolist()],
            "scale": [float(value) for value in scaler.scale_.tolist()],
        },
        "metadata": {
            "alpha": 0.0005,
        },
    }

    metadata_payload = {
        "model_name": export_payload["model_name"],
        "feature_count": len(DEPLOYABLE_FEATURE_COLUMNS),
        "feature_columns": DEPLOYABLE_FEATURE_COLUMNS,
        "split_strategy": "GroupShuffleSplit by source_id using deployable color features",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df["source_id"].nunique()) if "source_id" in train_df.columns else int(len(train_df)),
        "test_groups": int(test_df["source_id"].nunique()) if "source_id" in test_df.columns else int(len(test_df)),
        "train_metrics": train_metrics,
        "metrics": metrics,
        "target": "chlorine concentration",
        "unit": "mg/L",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MOBILE_APP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    MOBILE_MODEL_PATH.write_text(json.dumps(export_payload, indent=2))
    MOBILE_METADATA_PATH.write_text(json.dumps(metadata_payload, indent=2))
    MOBILE_APP_MODEL_PATH.write_text(json.dumps(export_payload, indent=2))

    print("Saved mobile model to:", MOBILE_MODEL_PATH)
    print("Saved mobile model metadata to:", MOBILE_METADATA_PATH)
    print("Copied mobile model into app at:", MOBILE_APP_MODEL_PATH)
    print("Holdout metrics:")
    print(json.dumps(metadata_payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
