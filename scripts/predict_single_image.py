import sys
import os
import json
import base64
import joblib
import pandas as pd

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import extract_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_artifacts")


def image_file_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def predict_image(image_path, model=None, scaler=None, feature_columns=None, metadata=None):
    if model is None:
        model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))

    if scaler is None:
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    if feature_columns is None:
        with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json"), "r") as f:
            feature_columns = json.load(f)

    if metadata is None:
        with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), "r") as f:
            metadata = json.load(f)

    image_base64 = image_file_to_base64(image_path)
    feature_dict = extract_features(image_base64)

    row_df = pd.DataFrame([feature_dict])
    row_df = row_df.reindex(columns=feature_columns, fill_value=0)

    X_scaled = scaler.transform(row_df.values)
    prediction = model.predict(X_scaled)[0]

    return {
        "predicted_concentration": round(float(prediction), 4),
        "unit": metadata.get("unit", "mg/L"),
        "model_name": metadata.get("model_name", "Unknown")
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_single_image.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_image(image_path)

    print("\nPrediction Result")
    print("=================")
    print(f"Model: {result['model_name']}")
    print(f"Predicted chlorine concentration: {result['predicted_concentration']:.4f} {result['unit']}")
