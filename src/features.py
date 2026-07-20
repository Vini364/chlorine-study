import os

from matplotlib import image
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc
import torch
torch.set_num_threads(1)
import json
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import rawpy
from pathlib import Path

# Dynamically locate the YOLO model
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "roi_with_augmentation.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)
IMAGES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "new_dataset_4.json")
# output_dir = "extracted_rois"
# os.makedirs(output_dir, exist_ok=True)


# --------------------------
# Helper function
# --------------------------
def is_file_path(image_input):
    """Determine if input is a file path or base64-encoded data."""
    if not isinstance(image_input, str):
        return False
    if len(image_input) > 1000:
        return False
    # Check for common file path indicators
    return (image_input.startswith('/') or image_input.startswith('.') or 
            '\\' in image_input or 
            image_input.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic', '.heif', '.dng', '.tiff')))

# --------------------------
# Decode base64 image
# --------------------------
def decode_image(image_input):
    """Decode image from either file path or base64-encoded data.
    
    Args:
        image_input: Either a file path (str) or base64-encoded image data (str)
    
    Returns:
        Image as RGB numpy array
    """
    if is_file_path(image_input):
        # Handle file path
        img_path = str(image_input)
        file_ext = os.path.splitext(img_path)[1].lower()
        if file_ext in {'.dng'}:
            img = rawpy.imread(img_path).postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
        elif file_ext in {'.heic', '.heif'}:
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
                with Image.open(img_path) as pil_image:
                    img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                raise ValueError(f"Failed to read HEIC image {img_path}") from exc
        else:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image from path: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Handle base64-encoded data
        import base64
        image_data = base64.b64decode(image_input)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from base64")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _largest_contour_bbox(mask, min_area=200):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def _expand_bbox(x, y, w, h, image_shape, x_pad=0.18, y_pad=0.18):
    height, width = image_shape[:2]
    pad_x = int(w * x_pad)
    pad_y = int(h * y_pad)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return x0, y0, x1, y1


def crop_test_tube_region(rgb_uint8):
    """
    Crop the analysis to the colored test-tube region.

    We first look for the pink liquid itself, since that is the signal that
    matters most for chlorine prediction. If that fails, we fall back to
    detecting the centered bottle body and trim the cap area.
    """
    height, width = rgb_uint8.shape[:2]
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)

    search_y0 = int(height * 0.18)
    search_y1 = int(height * 0.95)
    search_x0 = int(width * 0.18)
    search_x1 = int(width * 0.82)

    search_lab = lab[search_y0:search_y1, search_x0:search_x1]
    search_hsv = hsv[search_y0:search_y1, search_x0:search_x1]

    a_channel = search_lab[:, :, 1]
    hue = search_hsv[:, :, 0]
    saturation = search_hsv[:, :, 1]
    value = search_hsv[:, :, 2]

    pink_threshold = max(138, int(np.percentile(a_channel, 78)))
    pink_mask = (
        ((hue >= 150) | (hue <= 8)) &
        (a_channel >= pink_threshold) &
        (saturation >= 18) &
        (value <= 250)
    ).astype(np.uint8) * 255
    pink_mask = cv2.morphologyEx(
        pink_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )

    pink_bbox = _largest_contour_bbox(pink_mask, min_area=350)
    if pink_bbox is not None:
        x, y, w, h = pink_bbox
        x0, y0, x1, y1 = _expand_bbox(
            search_x0 + x,
            search_y0 + y,
            w,
            h,
            rgb_uint8.shape,
            x_pad=0.28,
            y_pad=0.30,
        )
        return rgb_uint8[y0:y1, x0:x1], pink_mask

    fallback = rgb_uint8[search_y0:search_y1, search_x0:search_x1]
    gray = cv2.cvtColor(fallback, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox = None
    best_score = None
    center_x = width / 2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        bbox_center_x = search_x0 + x + (w / 2)
        center_distance = abs(bbox_center_x - center_x) / width

        if aspect_ratio < 0.18 or aspect_ratio > 1.2:
            continue
        if center_distance > 0.22:
            continue

        score = area * (1 - center_distance)
        if best_score is None or score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    if best_bbox is not None:
        x, y, w, h = best_bbox
        x0, y0, x1, y1 = _expand_bbox(
            search_x0 + x,
            search_y0 + y + int(h * 0.18),
            w,
            int(h * 0.72),
            rgb_uint8.shape,
            x_pad=0.14,
            y_pad=0.12,
        )
        return rgb_uint8[y0:y1, x0:x1], pink_mask

    fallback_x0 = int(width * 0.28)
    fallback_x1 = int(width * 0.72)
    fallback_y0 = int(height * 0.30)
    fallback_y1 = int(height * 0.76)
    return rgb_uint8[fallback_y0:fallback_y1, fallback_x0:fallback_x1], pink_mask

# --------------------------
# Haralick features
# --------------------------
def extract_haralick_features(rgb_uint8):
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    gray = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    return {
        'haralick_contrast': graycoprops(glcm, 'contrast')[0, 0],
        'haralick_dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'haralick_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'haralick_energy': graycoprops(glcm, 'energy')[0, 0],
        'haralick_correlation': graycoprops(glcm, 'correlation')[0, 0],
        'haralick_ASM': graycoprops(glcm, 'ASM')[0, 0],
    }

# --------------------------
# Full features
# --------------------------
def extract_features(image_input, filename = None):
    """Extract features from an image provided as either a file path or base64 data.
    
    Args:
        image_input: Either a file path (str) or base64-encoded image data (str)
        filename: Optional filename for reference
    """
    rgb_uint8 = decode_image(image_input)
    features = {}
    with torch.no_grad():
        results = yolo_model(rgb_uint8, verbose = False)[0]
    if results.masks is None:
        binary_mask, pink_mask = crop_test_tube_region(rgb_uint8)
        rgb_float = binary_mask.astype(np.float32)
    else: 
        mask = results.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (results.orig_shape[1], results.orig_shape[0]))
        binary_mask = (mask > 0.5).astype(np.uint8)
        binary_mask = cv2.erode(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
        rgb_float = rgb_uint8[binary_mask == 1].astype(np.float32)


    R, G, B = rgb_float[:,0], rgb_float[:,1], rgb_float[:,2]

    features['mean_R'], features['mean_G'], features['mean_B'] = np.mean(R), np.mean(G), np.mean(B)
    features['std_R'], features['std_G'], features['std_B'] = np.std(R), np.std(G), np.std(B)
    total = features['mean_R'] + features['mean_G'] + features['mean_B']
    features['norm_r'] = features['mean_R']/total if total>0 else 0
    features['norm_g'] = features['mean_G']/total if total>0 else 0
    features['norm_b'] = features['mean_B']/total if total>0 else 0
    features['ratio_R_G'] = features['mean_R']/(features['mean_G']+1e-6)
    features['ratio_R_B'] = features['mean_R']/(features['mean_B']+1e-6)
    features['ratio_G_B'] = features['mean_G']/(features['mean_B']+1e-6)
    features['diff_R_G'] = features['mean_R']-features['mean_G']
    features['diff_R_B'] = features['mean_R']-features['mean_B']
    features['diff_G_B'] = features['mean_G']-features['mean_B']

    # hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    # features['mean_H'], features['mean_S'], features['mean_V'] = np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])
    # features['std_H'], features['std_S'], features['std_V'] = np.std(hsv[:,:,0]), np.std(hsv[:,:,1]), np.std(hsv[:,:,2])

    # lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    # features['mean_L'], features['mean_A'], features['mean_B_lab'] = np.mean(lab[:,:,0]), np.mean(lab[:,:,1]), np.mean(lab[:,:,2])
    # features['std_L'], features['std_A'], features['std_B_lab'] = np.std(lab[:,:,0]), np.std(lab[:,:,1]), np.std(lab[:,:,2])
    
    #For the neural net way:
    hsv_full = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    lab_full = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
        
    hsv_pixels = hsv_full[binary_mask == 1].astype(np.float32)
    H, S, V = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]

    lab_pixels = lab_full[binary_mask == 1].astype(np.float32)
    L, A, B_lab = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]

    features['mean_H'], features['mean_S'], features['mean_V'] = np.mean(H), np.mean(S), np.mean(V)
    features['std_H'], features['std_S'], features['std_V'] = np.std(H), np.std(S), np.std(V)

    features['mean_L'], features['mean_A'], features['mean_B_lab'] = np.mean(L), np.mean(A), np.mean(B_lab)
    features['std_L'], features['std_A'], features['std_B_lab'] = np.std(L), np.std(A), np.std(B_lab)

    features['brightness'] = np.mean(rgb_float)
    gray_est = (features['mean_R']+features['mean_G']+features['mean_B'])/3.0
    features['color_saturation'] = np.sqrt(
        (features['mean_R']-gray_est)**2 + (features['mean_G']-gray_est)**2 + (features['mean_B']-gray_est)**2
    )
    features['dominant_channel'] = int(np.argmax([features['mean_R'], features['mean_G'], features['mean_B']]))

    # Add background-normalized features
    features.update(compute_bg_normalized_features(rgb_uint8, binary_mask))

    # features.update(extract_haralick_features(rgb_uint8))

    if results.masks is not None and len(results.boxes) > 0:
        box = results.boxes.data[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box[:4])
        roi_crop = rgb_uint8[y1:y2, x1:x2]
        # if filename:
        #     roi_bgr = cv2.cvtColor(roi_crop, cv2.COLOR_RGB2BGR)
        #     save_name = f"roi_{os.path.splitext(filename)[0]}.jpg"
        #     cv2.imwrite(os.path.join(output_dir, save_name), roi_bgr)
        features.update(extract_haralick_features(roi_crop))
    else:
        # Emergency backup if YOLO misses
        features.update(extract_haralick_features(rgb_uint8))
    return features

# --------------------------
# Process dataset
# --------------------------
def process_dataset_records(data, output_csv=None):
    """Extract features from an in-memory dataset of image records."""
    print(f"Loaded {len(data)} images. Extracting features...")
    rows = []
    
    
    for idx, item in enumerate(data):
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(data)} images")
            gc.collect()

        image_base64 = item['image']
        concentration = float(item['concentration'])
        features = extract_features(image_base64, filename = item.get('filename'))
        if features is None:
            continue

        features['concentration'] = concentration
        features['is_augmented'] = item.get('is_augmented', False)
        features['source_id'] = item.get('source_id')
        features['filename'] = item.get('filename')
        rows.append(features)

        item['image'] = None  # Clear image data to save memory

    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved features to {output_csv}")

    return df


def process_augmented_data(input_json, output_csv):
    print(f"\nLoading augmented images from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)

    return process_dataset_records(data, output_csv=output_csv)

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    process_augmented_data("AimanSanskritiData.json", "features_data.csv")

def compute_bg_normalized_features(rgb_uint8, binary_mask):
    """
    Compute background-normalized features by comparing mask region vs background region.
    Returns a dict of normalized RGB and derived color features.
    """
    norm_features = {}
    
    # Get mask region pixels
    mask_pixels = rgb_uint8[binary_mask == 1].astype(np.float32)
    
    if len(mask_pixels) == 0:
        return norm_features
    
    # Get background region (inverted mask)
    bg_mask = (binary_mask == 0).astype(np.uint8)
    bg_pixels = rgb_uint8[bg_mask == 1].astype(np.float32)
    
    if len(bg_pixels) == 0:
        return norm_features
    
    # Compute mean RGB for mask and background
    mask_mean_r = np.mean(mask_pixels[:, 0])
    mask_mean_g = np.mean(mask_pixels[:, 1])
    mask_mean_b = np.mean(mask_pixels[:, 2])
    
    bg_mean_r = np.mean(bg_pixels[:, 0])
    bg_mean_g = np.mean(bg_pixels[:, 1])
    bg_mean_b = np.mean(bg_pixels[:, 2])
    
    # Normalized differences (mask - background)
    norm_features['norm_diff_R'] = mask_mean_r - bg_mean_r
    norm_features['norm_diff_G'] = mask_mean_g - bg_mean_g
    norm_features['norm_diff_B'] = mask_mean_b - bg_mean_b
    
    # Ratios (mask / background)
    norm_features['norm_ratio_R'] = mask_mean_r / (bg_mean_r + 1e-6)
    norm_features['norm_ratio_G'] = mask_mean_g / (bg_mean_g + 1e-6)
    norm_features['norm_ratio_B'] = mask_mean_b / (bg_mean_b + 1e-6)
    
    # Derived normalized features
    mask_total = mask_mean_r + mask_mean_g + mask_mean_b
    bg_total = bg_mean_r + bg_mean_g + bg_mean_b
    
    if mask_total > 0 and bg_total > 0:
        norm_features['norm_rg_diff'] = (mask_mean_r - mask_mean_g) - (bg_mean_r - bg_mean_g)
        norm_features['norm_rb_diff'] = (mask_mean_r - mask_mean_b) - (bg_mean_r - bg_mean_b)
        norm_features['norm_gb_diff'] = (mask_mean_g - mask_mean_b) - (bg_mean_g - bg_mean_b)
    
    return norm_features