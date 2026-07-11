import os
import json
import cv2
import numpy as np

from .features import decode_image, crop_test_tube_region  

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
INPUT_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "augmented_image_data.json")
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "images")
MASK_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def generate_masks():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print(f"Processing {len(data)} images...")

    for idx, item in enumerate(data):
        image_base64 = item["image"]

        rgb = decode_image(image_base64)

        roi, pink_mask = crop_test_tube_region(rgb)

        filename = f"img_{idx:04d}.png"
        cv2.imwrite(
            os.path.join(IMG_DIR, filename),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )

        cv2.imwrite(
            os.path.join(MASK_DIR, filename),
            pink_mask
        )

        if (idx + 1) % 50 == 0:
            print(f"Saved {idx + 1}/{len(data)}")

    print("✅ Done generating masks!")


if __name__ == "__main__":
    generate_masks()
