"""
Export a concise metrics summary for poster and presentation use.

Run this after:
  1. python -m src.pipeline
  2. python -m src.training_mobile
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "saved_artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "poster_metrics.json"


def load_json(path: Path):
    with path.open("r") as file:
        return json.load(file)


def main():
    model_metadata = load_json(ARTIFACTS_DIR / "model_metadata.json")
    mobile_metadata = load_json(ARTIFACTS_DIR / "mobile_model_metadata.json")
    candidate_results = load_json(ARTIFACTS_DIR / "candidate_results.json")
    selected_candidate = next(
        (item for item in candidate_results if item["model_name"] == model_metadata["model_name"]),
        None,
    )

    train_minus_holdout_r2 = model_metadata.get("train_minus_holdout_r2")
    holdout_to_train_rmse_ratio = model_metadata.get("holdout_to_train_rmse_ratio")
    holdout_to_train_mae_ratio = model_metadata.get("holdout_to_train_mae_ratio")

    if selected_candidate is not None:
        train_metrics = selected_candidate["train_original_metrics"]
        holdout_metrics = selected_candidate["holdout_metrics"]
        if train_minus_holdout_r2 is None:
            train_minus_holdout_r2 = train_metrics["r2"] - holdout_metrics["r2"]
        if holdout_to_train_rmse_ratio is None:
            holdout_to_train_rmse_ratio = holdout_metrics["rmse"] / max(train_metrics["rmse"], 1e-9)
        if holdout_to_train_mae_ratio is None:
            holdout_to_train_mae_ratio = holdout_metrics["mae"] / max(train_metrics["mae"], 1e-9)

    payload = {
        "selected_pipeline_model": {
            "model_name": model_metadata["model_name"],
            "train_original_r2": model_metadata["train_original_r2"],
            "cv_r2_mean": model_metadata["cv_r2_mean"],
            "cv_r2_std": model_metadata["cv_r2_std"],
            "holdout_r2": model_metadata["r2"],
            "train_original_rmse": model_metadata["train_original_rmse"],
            "holdout_rmse": model_metadata["rmse"],
            "train_original_mae": model_metadata["train_original_mae"],
            "holdout_mae": model_metadata["mae"],
            "overfit_severity": model_metadata["overfit_severity"],
            "train_minus_holdout_r2": train_minus_holdout_r2,
            "holdout_to_train_rmse_ratio": holdout_to_train_rmse_ratio,
            "holdout_to_train_mae_ratio": holdout_to_train_mae_ratio,
        },
        "mobile_model": {
            "model_name": mobile_metadata["model_name"],
            "train_r2": mobile_metadata["train_metrics"]["r2"],
            "test_r2": mobile_metadata["metrics"]["r2"],
            "train_rmse": mobile_metadata["train_metrics"]["rmse"],
            "test_rmse": mobile_metadata["metrics"]["rmse"],
            "train_mae": mobile_metadata["train_metrics"]["mae"],
            "test_mae": mobile_metadata["metrics"]["mae"],
        },
        "candidate_models": candidate_results,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved poster metrics to {OUTPUT_PATH}")
    print(json.dumps(payload["selected_pipeline_model"], indent=2))
    print(json.dumps(payload["mobile_model"], indent=2))


if __name__ == "__main__":
    main()
