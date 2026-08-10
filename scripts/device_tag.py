from src.features import is_file_path, _resolve_image_path
from PIL import Image
from PIL.ExifTags import TAGS
import os
import cv2
import exifread

def get_device_model_from_exif(image_path):
  if is_file_path(image_path):
    img_path = _resolve_image_path(image_path)
    file_ext = os.path.splitext(img_path)[1].lower()
    if file_ext in {'.heic', '.heif'}:
        try:
          from pillow_heif import register_heif_opener
          register_heif_opener()
          with Image.open(img_path) as pil_image:
              img = pil_image
              exif_data = img.getexif()
              if not exif_data:
                  return "unknown_no_exif"
              exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
              make = exif.get("Make", "").strip()
              model = exif.get("Model", "").strip()
              return f"{make} {model}".strip() or "unknown_no_exif"
        except Exception as e:
          print(f"ERROR on {image_path}: {type(e).__name__}: {e}")  # <-- see the real error
          return "unknown_no_exif"
    elif file_ext in {'.dng'}:
        try:
          with open(img_path, "rb") as f:
              tags = exifread.process_file(f, details=False)
          make = str(tags.get("Image Make", "")).strip()
          model = str(tags.get("Image Model", "")).strip()
          if not model:
            model = str(tags.get("Image Tag 0xC614", "")).strip()
          return f"{make} {model}".strip() or "unknown_no_exif"
        except Exception as e:
          print(f"ERROR on {image_path}: {type(e).__name__}: {e}")
          return "unknown_error"
    else:
      img = cv2.imread(img_path)
      if img is None:
          raise ValueError(f"Failed to read image from path: {img_path}")
      exif_data = cv2.imencode('.jpg', img)[1].tobytes()
      if not exif_data:
          return "unknown_no_exif"
      try:
          from io import BytesIO
          pil_image = Image.open(BytesIO(exif_data))
          exif = pil_image._getexif()
          if not exif:
              return "unknown_no_exif"
          exif = {TAGS.get(k, k): v for k, v in exif.items()}
          make = exif.get("Make", "").strip()
          model = exif.get("Model", "").strip()
          return f"{make} {model}".strip() or "unknown_no_exif"
      except Exception as e:
          print(f"ERROR on {image_path}: {type(e).__name__}: {e}")
          return "unknown_no_exif"

def model_from_exif(image_path):
    tag = get_device_model_from_exif(image_path).upper()
    if "APPLE" in tag or "IPHONE" in tag:
        return "iphone"
    elif "SAMSUNG" in tag or "ANDROID" in tag or "SM-" in tag:
        return "android"
    else:   
        return "unknown"


def get_time_from_exif(image_path):
  if is_file_path(image_path):
    img_path = _resolve_image_path(image_path)
    file_ext = os.path.splitext(img_path)[1].lower()
    if file_ext in {'.heic', '.heif'}:
        try:
          from pillow_heif import register_heif_opener
          register_heif_opener()
          with Image.open(img_path) as pil_image:
              img = pil_image
              exif_data = img.getexif()
              if not exif_data:
                  return "unknown_no_exif"
              exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
              exif_date = exif.get("DateTime", "").strip().split(" ")[0]
              return f"{exif_date}" or "unknown_no_exif"
        except Exception as e:
          print(f"ERROR on {image_path}: {type(e).__name__}: {e}")  # <-- see the real error
          return "unknown_no_exif"
    elif file_ext in {'.dng'}:
        try:
          with open(img_path, "rb") as f:
              tags = exifread.process_file(f, details=False)
          date_time = str(tags.get("Image DateTime", "")).strip()
          if not date_time:
            date_time = (os.path.splitext(os.path.basename(img_path))[0]).split("_")[1]
          return f"{date_time}".strip() or "unknown_no_exif"
        except Exception as e:
          print(f"ERROR on {image_path}: {type(e).__name__}: {e}")
          return "unknown_error"
    else:
      img = cv2.imread(img_path)
      if img is None:
          raise ValueError(f"Failed to read image from path: {img_path}")
      exif_data = cv2.imencode('.jpg', img)[1].tobytes()
      if not exif_data:
          return "unknown_no_exif"
      try:
          from io import BytesIO
          pil_image = Image.open(BytesIO(exif_data))
          exif = pil_image._getexif()
          if not exif:
              return "unknown_no_exif"
          exif = {TAGS.get(k, k): v for k, v in exif.items()}
          exif_date = exif.get("DateTime", "").strip().split(" ")[0]
          return f"{exif_date}" or "unknown_no_exif"
      except Exception as e:
          print(f"ERROR on {image_path}: {type(e).__name__}: {e}")
          return "unknown_no_exif"


