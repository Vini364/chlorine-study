from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import tempfile
import os
import json
import logging
from datetime import datetime
from scripts.predict_single_image import predict_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chlorine Concentration Detection API",
    description="API for predicting chlorine concentration from test tube images",
    version="1.0.0"
)

# Add CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
HEIF_EXTENSIONS = {".heic", ".heif"}

# Calculate default artifacts directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_artifacts")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)
EXPECTED_MODEL_NAME = os.getenv("EXPECTED_MODEL_NAME", "ElasticNet Regression")

# Global variables for cached models
model = None
scaler = None
feature_columns = None
model_metadata = None


def save_normalized_upload(contents: bytes, file_ext: str) -> str:
    """
    Persist an uploaded image to a temporary path.

    HEIC/HEIF uploads are converted to JPEG so the existing OpenCV feature
    extraction pipeline can decode them reliably.
    """
    normalized_ext = file_ext.lower()

    if normalized_ext in HEIF_EXTENSIONS:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise RuntimeError(
                "HEIC/HEIF support is not installed. Add pillow-heif to the environment."
            ) from exc

        register_heif_opener()

        with Image.open(BytesIO(contents)) as image:
            rgb_image = image.convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                rgb_image.save(tmp, format="JPEG", quality=95)
                return tmp.name

    suffix = normalized_ext or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        return tmp.name

def load_model_artifacts():
    """Load model artifacts once at startup"""
    global model, scaler, feature_columns, model_metadata

    if model is not None:
        return  # Already loaded

    try:
        import joblib

        model_path = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        features_path = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
        metadata_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {ARTIFACTS_DIR}")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        with open(features_path, "r") as f:
            feature_columns = json.load(f)

        with open(metadata_path, "r") as f:
            model_metadata = json.load(f)

        loaded_model_name = model_metadata.get("model_name")
        if loaded_model_name != EXPECTED_MODEL_NAME:
            raise RuntimeError(
                f"Expected model '{EXPECTED_MODEL_NAME}' but found '{loaded_model_name}'. "
                "Retrain or replace the saved artifacts before starting the app."
            )

        logger.info("Model artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_model_artifacts()

@app.get("/", include_in_schema=False)
def root():
    """Basic API root response."""
    return {
        "message": "Chlorine Concentration Detection API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "mobile_app": "mobile/",
    }

@app.get("/api/v1/health")
def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/api/v1/info")
def model_info():
    """Get model metadata and API information"""
    if model_metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model": model_metadata,
        "api_version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "info": "/api/v1/info",
            "predict": "/api/v1/predict"
        }
    }

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict chlorine concentration from an image.

    The image should show a test tube with DPD reagent.
    Response contains predicted concentration in mg/L.
    """
    try:
        # Validate model is loaded
        if model is None:
            logger.error("Model not loaded during prediction request")
            raise HTTPException(status_code=500, detail="Model not initialized. Please try again.")

        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Read file and validate size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Save to a normalized temporary file
        temp_path = save_normalized_upload(contents, file_ext)

        logger.info(f"Processing image: {file.filename}")

        try:
            # Make prediction using cached model
            result = predict_image(
                temp_path,
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
                metadata=model_metadata,
            )
            logger.info(f"Prediction successful: {result['predicted_concentration']} mg/L")

            return {
                "success": True,
                "predicted_concentration": result['predicted_concentration'],
                "unit": result['unit'],
                "model_name": result['model_name'],
                "timestamp": datetime.utcnow().isoformat()
            }
        except ValueError as e:
            logger.error(f"Image processing error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Could not process image: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to make prediction. Please check the image and try again."
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Keep old endpoints for backward compatibility
@app.get("/health")
def health():
    """Legacy health endpoint - redirects to new endpoint"""
    return health_check()

@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    """Legacy predict endpoint - redirects to new endpoint"""
    return await predict(file)
