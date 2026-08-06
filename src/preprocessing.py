"""
Image augmentation for chlorine test tube dataset.
"""
import cv2 as cv
import json
import base64
import numpy as np
import os
from PIL import Image, ImageEnhance
import io
import albumentations as A
import rawpy
from pillow_heif import register_heif_opener
register_heif_opener()
MAX_DIM = 800
transform = A.Compose([
    A.Rotate(limit=45, border_mode=cv.BORDER_REFLECT, p=0.2),
    A.HorizontalFlip(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=0,
        border_mode=cv.BORDER_REFLECT,
        p=0.5
    ),
    # A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
])

def is_file_path(image_input):
    """Determine if input is a file path or base64-encoded data."""
    if not isinstance(image_input, str):
        return False
    # Check for common file path indicators
    return (image_input.startswith('/') or image_input.startswith('.') or 
            '\\' in image_input or 
            image_input.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic', '.heif', '.dng', '.tiff')))

def prepare_image(image_input):
    if is_file_path(image_input):
            format = os.path.splitext(image_input)[1].lower()
            if format in {'.dng'}:
                img = rawpy.imread(image_input).postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            elif format in {'.heic', '.heif'}:
                try:
                    with Image.open(image_input) as pil_image:
                        img = cv.cvtColor(np.array(pil_image.convert("RGB")), cv.COLOR_RGB2BGR)
                except Exception as exc:
                    raise ValueError(f"Failed to read HEIC image {image_input}") from exc
            else:
                img = cv.imread(image_input)
                if img is None:
                    raise ValueError(f"Failed to read image from path: {image_input}")
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        # It's base64-encoded
        image_data = base64.b64decode(image_input)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img = Image.fromarray(img)
    img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS) 
    return img


def transform_image(img):
    img_array = np.array(img, dtype=np.uint8)

    img_array = transform(image=img_array)['image'] 
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=92)
    augmented_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return augmented_base64
    


# def augment_image(image_input, augmentation_type=None, max_dim = 800):
#     """Augment a single image from either file path or base64-encoded data.
    
#     Args:
#         image_input: Either a file path (str) or base64-encoded image data (str)
#         augmentation_type: Optional specific augmentation to apply
#         max_dim: Maximum dimension for resizing
#     """
    # Determine if input is a file path or base64

    #PREVIOUS:

    # img_array = np.array(img, dtype=np.float32)

    # # Choose random augmentation type
    # aug_type = augmentation_type or random.choice([
    #     'rotation', 'brightness', 'contrast', 'color_shift', 'noise', 'gamma'
    # ])

    # # Apply augmentation
    # if aug_type == 'rotation':
    #     angle = random.uniform(-15, 15)
    #     img = Image.fromarray(np.uint8(img_array)).rotate(angle, expand=False)
    #     img_array = np.array(img, dtype=np.float32)
    # elif aug_type == 'brightness':
    #     delta = random.uniform(-18, 18)
    #     img_array = np.clip(img_array + delta, 0, 255)
    # elif aug_type == 'contrast':
    #     factor = random.uniform(0.85, 1.15)
    #     img = Image.fromarray(np.uint8(img_array))
    #     enhancer = ImageEnhance.Contrast(img)
    #     img = enhancer.enhance(factor)
    #     img_array = np.array(img, dtype=np.float32)
    # elif aug_type == 'color_shift':
    #     shift_channel = random.choice([0, 1, 2])
    #     delta = random.uniform(-30, 30)
    #     img_array[:, :, shift_channel] = np.clip(
    #         img_array[:, :, shift_channel] + delta, 0, 255
    #     )
    # elif aug_type == 'noise':
    #     if random.random() < 0.5:
    #         noise = np.random.normal(0, 2, img_array.shape)
    #         img_array = np.clip(img_array + noise, 0, 255)
    # elif aug_type == 'gamma':
    #     if random.random() < 0.5:
    #         gamma = random.uniform(0.9, 1.1)
    #         img_array = 255.0 * np.power(img_array / 255.0, gamma)
    #         img_array = np.clip(img_array, 0, 255)

    # Convert back to image and encode as base64
    # img_array = np.uint8(np.clip(img_array, 0, 255))
    # img = Image.fromarray(img_array)
    # buffer = io.BytesIO()
    # img.save(buffer, format='JPEG', quality=95)
    # augmented_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # return augmented_base64

# def augment_dataset(input_json, output_json, augmentations_per_image=2, save_augmented_images=False, output_image_dir=None):
#     """Augment all images in dataset and save to output JSON.
    
#     Args:
#         input_json: Path to input JSON file
#         output_json: Path to output JSON file  
#         augmentations_per_image: Number of augmentations per image
#         save_augmented_images: If True, save augmented images to disk and store paths
#         output_image_dir: Directory to save augmented images (required if save_augmented_images=True)
#     """
#     print(f"Loading dataset from {input_json}...")
#     with open(input_json, 'r') as f:
#         dataset = json.load(f)
#     print(f"Loaded {len(dataset)} images")

#     augmented_dataset = []

#     # Add original images
#     for i, sample in enumerate(dataset):
#         augmented_dataset.append({
#             'image': sample['image'],
#             'concentration': sample['concentration'],
#             'is_augmented': False,
#             'filename': sample.get('filename')
#         })
#         if (i + 1) % 50 == 0:
#             print(f"  Processed {i + 1}/{len(dataset)} originals")

#     # Create augmented versions
#     total = len(dataset)
#     if save_augmented_images and output_image_dir:
#         os.makedirs(output_image_dir, exist_ok=True)
    
#     for i, sample in enumerate(dataset):
#         original_image = prepare_image(sample['image'])
#         for j in range(augmentations_per_image):
#             try:
#                 augmented_img = transform_image(original_image)
#                 if save_augmented_images and output_image_dir:
#                     # Save augmented image to disk
#                     base_filename = os.path.splitext(sample.get('filename', f'sample_{i}'))[0]
#                     img_filename = f"{base_filename}_aug_{j}.jpg"
#                     img_path = os.path.join(output_image_dir, img_filename)
#                     img_pil = Image.fromarray((augmented_img * 255).astype(np.uint8)) if isinstance(augmented_img, np.ndarray) else Image.open(io.BytesIO(base64.b64decode(augmented_img)))
#                     img_pil.save(img_path, quality=92)
#                     image_field = img_path
#                 else:
#                     # Keep as base64
#                     image_field = augmented_img
                
#                 augmented_dataset.append({
#                     'image': image_field,
#                     'concentration': sample['concentration'],
#                     'is_augmented': True,
#                     'filename': sample.get('filename')
#                 })
#             except Exception as e:
#                 print(f"Warning: Failed to augment image {i}: {str(e)}")
#         if (i + 1) % 50 == 0:
#             print(f"  Created augmentations for {i + 1}/{total} images")

#     print(f"\nSaving augmented dataset ({len(augmented_dataset)} images) to {output_json}...")
#     with open(output_json, 'w') as f:
#         json.dump(augmented_dataset, f)
#     print(f"✓ Augmentation complete: {len(augmented_dataset)} total images")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         input_file = sys.argv[1]
#         output_file = sys.argv[2] if len(sys.argv) > 2 else 'augmented_image_data.json'
#         augmentations = int(sys.argv[3]) if len(sys.argv) > 3 else 2
#         augment_dataset(input_file, output_file, augmentations)
#     else:
#         print("Usage: python augment_images.py <input_json> [output_json] [augmentations_per_image]")
