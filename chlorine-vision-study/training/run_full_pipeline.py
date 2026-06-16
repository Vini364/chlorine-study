"""
End-to-end training pipeline with a more realistic phone-photo evaluation flow.

Key safeguards:
1. Split original images before augmentation.
2. Augment only the training subset.
3. Evaluate on untouched holdout photos.
4. Optionally evaluate on an external phone validation dataset if provided.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from training.augment_images import augment_image
from training.extract_features import process_dataset_records


TRAINING_DIR = "training"
ARTIFACTS_DIR = os.path.join(TRAINING_DIR, "saved_artifacts")
EXTERNAL_PHONE_VALIDATION_PATH = os.path.join(TRAINING_DIR, "phone_validation_dataset.json")


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def choose_input_dataset():
    candidates = [
        os.path.join(TRAINING_DIR, "new_dataset.json"),
        os.path.join(TRAINING_DIR, "processed_image_data.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("No input dataset found. Expected training/new_dataset.json or training/processed_image_data.json")


def load_dataset_records(path):
    with open(path, "r") as f:
        dataset = json.load(f)

    records = []
    for index, item in enumerate(dataset):
        filename = item.get("filename", f"sample_{index}.jpg")
        source_id = item.get("source_id", filename)
        records.append({
            "image": item["image"],
            "concentration": float(item["concentration"]),
            "filename": filename,
            "source_id": source_id,
            "is_augmented": bool(item.get("is_augmented", False)),
        })

    return records


def make_stratify_labels(values, max_bins=5):
    unique_count = len(np.unique(values))
    if unique_count < 2:
        return None

    bin_count = min(max_bins, unique_count)
    return pd.qcut(values, q=bin_count, duplicates="drop").astype(str)


def split_original_records(records, test_size=0.2, random_state=42):
    y = np.array([record["concentration"] for record in records], dtype=float)
    stratify_labels = make_stratify_labels(y)
    indices = np.arange(len(records))

    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    train_records = [records[index] for index in train_idx]
    holdout_records = [records[index] for index in holdout_idx]
    return train_records, holdout_records


def augment_training_records(records, augmentations_per_image=2):
    augmented_records = []

    for index, sample in enumerate(records):
        original_record = dict(sample)
        original_record["is_augmented"] = False
        augmented_records.append(original_record)

        for _ in range(augmentations_per_image):
            augmented_records.append({
                "image": augment_image(sample["image"]),
                "concentration": sample["concentration"],
                "filename": sample["filename"],
                "source_id": sample["source_id"],
                "is_augmented": True,
            })

        if (index + 1) % 25 == 0:
            print(f"Prepared training augmentations for {index + 1}/{len(records)} originals")

    return augmented_records


def evaluate_predictions(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def classify_overfit_gap(train_metrics, holdout_metrics):
    r2_gap = train_metrics["r2"] - holdout_metrics["r2"]
    rmse_ratio = holdout_metrics["rmse"] / max(train_metrics["rmse"], 1e-9)
    mae_ratio = holdout_metrics["mae"] / max(train_metrics["mae"], 1e-9)

    if r2_gap > 0.12 or rmse_ratio > 4.5 or mae_ratio > 4.5:
        return "severe"
    if r2_gap > 0.05 or rmse_ratio > 2.5 or mae_ratio > 2.5:
        return "moderate"
    return "mild"


def compute_gap_summary(train_metrics, holdout_metrics):
    return {
        "train_minus_holdout_r2": float(train_metrics["r2"] - holdout_metrics["r2"]),
        "holdout_to_train_rmse_ratio": float(holdout_metrics["rmse"] / max(train_metrics["rmse"], 1e-9)),
        "holdout_to_train_mae_ratio": float(holdout_metrics["mae"] / max(train_metrics["mae"], 1e-9)),
    }


def get_model_search_space(random_state=42):
    return {
        # "Ridge Regression": (
        #     Ridge(),
        #     {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]},
        # ),
        # "Lasso Regression": (
        #     Lasso(max_iter=20000),
        #     {"alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]},
        # ),
        # "ElasticNet Regression": (
        #     ElasticNet(max_iter=20000),
        #     {
        #         "alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        #         "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        #     },
        # ),
        # "Random Forest Regression": (
        #     RandomForestRegressor(random_state=random_state),
        #     {
        #         "n_estimators": [200, 400],
        #         "max_depth": [None, 6, 10],
        #         "min_samples_leaf": [1, 2, 4],
        #     },
        # ),
        # "Gradient Boosting Regression": (
        #     GradientBoostingRegressor(random_state=random_state),
        #     {
        #         "n_estimators": [100, 200],
        #         "learning_rate": [0.03, 0.05, 0.1],
        #         "max_depth": [2, 3],
        #     },
        # ),
        "SVR": (
            SVR(),
            {
                "C": [0.1, 1.0, 10.0, 100.0],
                "epsilon": [0.01, 0.1, 0.5],
                "kernel": ["rbf", "linear"],
            },
        ),
        # "Extra Trees Regression": (
        #     ExtraTreesRegressor(random_state=random_state),
        #     {
        #         "n_estimators": [200, 400],
        #         "max_depth": [None, 6, 10],
        #         "min_samples_leaf": [1, 2, 4],
        #     },
        # ),
        # "XGBoost Regression": (
        #     XGBRegressor(random_state=random_state, verbosity=0),
        #     {
        #         "n_estimators": [100, 200],
        #         "learning_rate": [0.03, 0.05, 0.1],
        #         "max_depth": [3, 5],
        #         "subsample": [0.8, 1.0],
        #     },
        # ),
        # "LightGBM Regression": (
        #     LGBMRegressor(random_state=random_state, verbosity=-1),
        #     {
        #         "n_estimators": [100, 200],
        #         "learning_rate": [0.03, 0.05, 0.1],
        #         "max_depth": [-1, 6, 10],
        #         "num_leaves": [31, 63],
        #     },
        # ),
    }


def train_candidates(
    X_train,
    y_train,
    train_groups,
    X_train_original,
    y_train_original,
    X_holdout,
    y_holdout,
    feature_columns,
    random_state=42,
):
    group_count = len(np.unique(train_groups))
    cv_splits = max(3, min(5, group_count))
    cv = GroupKFold(n_splits=cv_splits)

    results = []

    for model_name, (estimator, param_grid) in get_model_search_space(random_state=random_state).items():
        print(f"\nTuning {model_name}...")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid={f"model__{key}": value for key, value in param_grid.items()},
            scoring="r2",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train, groups=train_groups)
        feature_importance = permutation_importance(search.best_estimator_, X_holdout, y_holdout, n_repeats = 5, random_state = 42)
        importance_df = pd.DataFrame({
            'Feature': feature_columns.columns,
            'Importance_Mean': feature_importance.importances_mean,
            'Importance_Std': feature_importance.importances_std
        }).sort_values(by='Importance_Mean', ascending=False)

        train_original_predictions = search.best_estimator_.predict(X_train_original)
        train_original_metrics = evaluate_predictions(y_train_original, train_original_predictions)
        holdout_predictions = search.best_estimator_.predict(X_holdout)
        holdout_metrics = evaluate_predictions(y_holdout, holdout_predictions)
        gap_summary = compute_gap_summary(train_original_metrics, holdout_metrics)
        overfit_severity = classify_overfit_gap(train_original_metrics, holdout_metrics)

        for idx, row in importance_df.head().iterarows(5):
            print(f"{row['Feature']} : {row['Importance_Mean']:.4f}")

        print(f"  Best params: {search.best_params_}")
        print(f"  CV R²: {search.best_score_:.4f}")
        print(f"  Train-original R²: {train_original_metrics['r2']:.4f}")
        print(f"  Holdout R²: {holdout_metrics['r2']:.4f}")
        print(f"  Holdout RMSE: {holdout_metrics['rmse']:.4f}")
        print(f"  Holdout MAE: {holdout_metrics['mae']:.4f}")
        print(f"  Train-holdout R² gap: {gap_summary['train_minus_holdout_r2']:.4f}")
        print(f"  Overfitting severity: {overfit_severity}")

        results.append({
            "model_name": model_name,
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_r2_mean": float(search.best_score_),
            "cv_r2_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "train_original_metrics": train_original_metrics,
            "holdout_metrics": holdout_metrics,
            "gap_summary": gap_summary,
            "overfit_severity": overfit_severity,
        })

    severity_rank = {"mild": 0, "moderate": 1, "severe": 2}
    results.sort(
        key=lambda item: (
            severity_rank.get(item["overfit_severity"], 99),
            item["gap_summary"]["train_minus_holdout_r2"],
            -item["holdout_metrics"]["r2"],
            item["holdout_metrics"]["rmse"],
        ),
    )
    return results


def evaluate_external_phone_validation(best_estimator, feature_columns):
    if not os.path.exists(EXTERNAL_PHONE_VALIDATION_PATH):
        print("\nNo external phone validation dataset found. Skipping that step.")
        return None

    print("\nEvaluating on external phone validation dataset...")
    phone_records = load_dataset_records(EXTERNAL_PHONE_VALIDATION_PATH)
    phone_df = process_dataset_records(phone_records)

    X_phone = phone_df.reindex(columns=feature_columns, fill_value=0).values
    y_phone = phone_df["concentration"].values
    predictions = best_estimator.predict(X_phone)

    metrics = evaluate_predictions(y_phone, predictions)
    metrics["samples"] = int(len(y_phone))

    print(f"  External phone R²: {metrics['r2']:.4f}")
    print(f"  External phone RMSE: {metrics['rmse']:.4f}")
    print(f"  External phone MAE: {metrics['mae']:.4f}")

    return metrics


def run_pipeline(augmentations_per_image=2, random_state=42):
    print_section("CHLORINE CONCENTRATION PREDICTION PIPELINE")

    input_file = choose_input_dataset()
    print(f"Using input file: {input_file}")

    print_section("STEP 1: SPLIT ORIGINAL PHONE PHOTOS BEFORE AUGMENTATION")
    original_records = load_dataset_records(input_file)
    train_originals, holdout_originals = split_original_records(
        original_records,
        test_size=0.2,
        random_state=random_state,
    )
    print(f"Original images: {len(original_records)}")
    print(f"Training originals: {len(train_originals)}")
    print(f"Untouched holdout originals: {len(holdout_originals)}")

    print_section("STEP 2: AUGMENT TRAINING IMAGES ONLY")
    train_records = augment_training_records(
        train_originals,
        augmentations_per_image=augmentations_per_image,
    )
    print(f"Training records after augmentation: {len(train_records)}")

    os.makedirs(TRAINING_DIR, exist_ok=True)
    with open(os.path.join(TRAINING_DIR, "augmented_image_data.json"), "w") as f:
        json.dump(train_records, f)
    with open(os.path.join(TRAINING_DIR, "phone_holdout_dataset.json"), "w") as f:
        json.dump(holdout_originals, f)

    print_section("STEP 3: FEATURE EXTRACTION")
    train_df = process_dataset_records(
        train_records,
        output_csv=os.path.join(TRAINING_DIR, "features_data.csv"),
    )
    holdout_df = process_dataset_records(holdout_originals)

    feature_columns = [
        column for column in train_df.columns
        if column not in {"concentration", "is_augmented", "source_id", "filename"}
    ]

    X_train = train_df[feature_columns].values
    y_train = train_df["concentration"].values
    train_groups = train_df["source_id"].fillna(train_df["filename"]).values
    train_original_df = train_df[train_df["is_augmented"] == False].copy()
    X_train_original = train_original_df[feature_columns].values
    y_train_original = train_original_df["concentration"].values

    X_holdout = holdout_df.reindex(columns=feature_columns, fill_value=0).values
    y_holdout = holdout_df["concentration"].values

    print(f"Features used for training: {len(feature_columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Holdout samples: {len(X_holdout)}")

    print_section("STEP 4: MODEL SEARCH WITH GROUP-AWARE CROSS-VALIDATION")
    candidate_results = train_candidates(
        X_train,
        y_train,
        train_groups,
        X_train_original,
        y_train_original,
        X_holdout,
        y_holdout,
        feature_columns,
        random_state=random_state,
    )

    best_result = candidate_results[0]
    best_estimator = best_result["best_estimator"]
    best_model = best_estimator.named_steps["model"]
    best_scaler = best_estimator.named_steps["scaler"]

    print("\nBest model based on untouched holdout R²:")
    print(f"  {best_result['model_name']}")
    print(f"  Holdout R²: {best_result['holdout_metrics']['r2']:.4f}")
    print(f"  Holdout RMSE: {best_result['holdout_metrics']['rmse']:.4f}")
    print(f"  Holdout MAE: {best_result['holdout_metrics']['mae']:.4f}")

    print_section("STEP 5: OPTIONAL EXTERNAL PHONE VALIDATION")
    external_phone_metrics = evaluate_external_phone_validation(best_estimator, feature_columns)

    print_section("STEP 6: SAVE ARTIFACTS")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
    joblib.dump(best_scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    metadata = {
        "model_name": best_result["model_name"],
        "best_params": best_result["best_params"],
        "augmentations_per_image": augmentations_per_image,
        "split_strategy": "Split original phone photos before augmentation",
        "training_original_samples": len(train_originals),
        "training_samples_with_augmentation": len(train_records),
        "holdout_original_samples": len(holdout_originals),
        "cv_r2_mean": best_result["cv_r2_mean"],
        "cv_r2_std": best_result["cv_r2_std"],
        "train_original_r2": best_result["train_original_metrics"]["r2"],
        "train_original_rmse": best_result["train_original_metrics"]["rmse"],
        "train_original_mae": best_result["train_original_metrics"]["mae"],
        "train_minus_holdout_r2": best_result["gap_summary"]["train_minus_holdout_r2"],
        "holdout_to_train_rmse_ratio": best_result["gap_summary"]["holdout_to_train_rmse_ratio"],
        "holdout_to_train_mae_ratio": best_result["gap_summary"]["holdout_to_train_mae_ratio"],
        "overfit_severity": best_result["overfit_severity"],
        "r2": best_result["holdout_metrics"]["r2"],
        "rmse": best_result["holdout_metrics"]["rmse"],
        "mae": best_result["holdout_metrics"]["mae"],
        "target": "chlorine concentration",
        "unit": "mg/L",
    }

    if external_phone_metrics is not None:
        metadata["external_phone_validation"] = external_phone_metrics

    with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "candidate_results.json"), "w") as f:
        json.dump([
            {
                "model_name": result["model_name"],
                "best_params": result["best_params"],
                "cv_r2_mean": result["cv_r2_mean"],
                "cv_r2_std": result["cv_r2_std"],
                "train_original_metrics": result["train_original_metrics"],
                "holdout_metrics": result["holdout_metrics"],
                "gap_summary": result["gap_summary"],
                "overfit_severity": result["overfit_severity"],
            }
            for result in candidate_results
        ], f, indent=2)

    print("Saved model artifacts:")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'best_model.pkl')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'scaler.pkl')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'feature_columns.json')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'model_metadata.json')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'candidate_results.json')}")

    print_section("PIPELINE COMPLETED SUCCESSFULLY")
    return True


if __name__ == "__main__":
    run_pipeline()
