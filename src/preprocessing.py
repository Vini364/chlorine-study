"""
Image augmentation for chlorine test tube dataset.
"""
import cv2 as cv
import json
import base64
import numpy as np
from PIL import Image, ImageEnhance
import io
import random
import albumentations as A
def augment_image(image_base64, augmentation_type=None, max_dim = 800):
    """Augment a single base64-encoded image"""
    # Decode base64
    image_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img.thumbnail((max_dim, max_dim), Image.LANCZOS) 
    
    # 2. Convert to uint8 NumPy array (0-255)
    # CRITICAL: Keep as uint8 because Albumentations/OpenCV expect 0-255 integers
    img_array = np.array(img, dtype=np.uint8)

    transform = A.Compose([
        A.Rotate(limit=45, border_mode=cv.BORDER_REFLECT, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=0,
            border_mode=cv.BORDER_REFLECT,
            p=0.5
        ),
        # A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
    ])

    img_array = transform(image=img_array)['image'] 

    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=92)
    augmented_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

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
    return augmented_base64

def augment_dataset(input_json, output_json, augmentations_per_image=2):
    """Augment all images in dataset and save to output JSON"""
    print(f"Loading dataset from {input_json}...")
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} images")

    augmented_dataset = []

    # Add original images
    for i, sample in enumerate(dataset):
        augmented_dataset.append({
            'image': sample['image'],
            'concentration': sample['concentration'],
            'is_augmented': False
        })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} originals")

    # Create augmented versions
    total = len(dataset)
    for i, sample in enumerate(dataset):
        for j in range(augmentations_per_image):
            try:
                augmented_img = augment_image(sample['image'])
                augmented_dataset.append({
                    'image': augmented_img,
                    'concentration': sample['concentration'],
                    'is_augmented': True
                })
            except Exception as e:
                print(f"Warning: Failed to augment image {i}: {str(e)}")
        if (i + 1) % 50 == 0:
            print(f"  Created augmentations for {i + 1}/{total} images")

    print(f"\nSaving augmented dataset ({len(augmented_dataset)} images) to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(augmented_dataset, f)
    print(f"✓ Augmentation complete: {len(augmented_dataset)} total images")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'augmented_image_data.json'
        augmentations = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        augment_dataset(input_file, output_file, augmentations)
    else:
        print("Usage: python augment_images.py <input_json> [output_json] [augmentations_per_image]")
