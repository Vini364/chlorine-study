"""
Image augmentation for chlorine test tube dataset.
"""

import json
import base64
import numpy as np
from PIL import Image, ImageEnhance
import io
import random
import albumentations as A
def augment_image(image_base64, augmentation_type=None):
    """Augment a single base64-encoded image"""
    # Decode base64
    image_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # # 2. Convert to uint8 NumPy array (0-255)
    # # CRITICAL: Keep as uint8 because Albumentations/OpenCV expect 0-255 integers
    # img_array = np.array(img, dtype=np.uint8)

    # # 3. Choose random augmentation type
    # aug_type = augmentation_type or random.choice([
    #     'rotation', 'noise', 'blur', 'horizontal_flip'
    # ])

    # # 4. Define Albumentations blocks
    # # if aug_type == 'rotation':
    # #     transform = A.Rotate(limit=(-15, 15), p=1.0)
    # #     img_array = transform(image=img_array)['image']
        
    # # elif aug_type == 'brightness':
    # #     transform = A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=0, p=1.0)
    # #     img_array = transform(image=img_array)['image']
        
    # # elif aug_type == 'contrast':
    # #     transform = A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.15, 0.15), p=1.0)
    # #     img_array = transform(image=img_array)['image']
        
    # # elif aug_type == 'samsung_vivid':
    # #     # Safely juices up saturation and contrast in HSV space without breaking RGB channel reality
    # #     transform = A.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.4, hue=0.02, p=1.0)
    # #     img_array = transform(image=img_array)['image']
        
    # # elif aug_type == 'pixel_cool':
    # #     # Drops brightness by 5%, boosts contrast by 25%, drops saturation by 10%, and shifts hue
    # #     transform = A.ColorJitter(
    # #         brightness=(0.95, 0.95),  # 1.0 - 0.05 = 0.95 (Dimmed)
    # #         contrast=(1.25, 1.25),    # 1.0 + 0.25 = 1.25 (Sharpened)
    # #         saturation=(0.90, 0.90),  # 1.0 - 0.10 = 0.90 (Desaturated)
    # #         hue=(-0.02, -0.02),       # Hue stays additive, allowing negative boundaries
    # #         p=1.0
    # #     )
    # #     img_array = transform(image=img_array)['image']
        
    # # elif aug_type == 'noise':
    # #     # Replaced var_limit with std_limit for modern Albumentations versions
    # #     transform = A.GaussNoise(std_limit=(10.0, 30.0), p=1.0)
    # #     img_array = transform(image=img_array)['image']
        
    # # elif aug_type == 'gamma':
    # #     transform = A.RandomGamma(gamma_limit=(80, 120), p=1.0)
    # #     img_array = transform(image=img_array)['image']
    # if aug_type == 'rotation':
    #     transform = A.Rotate(limit=(-10, 10), p=1.0)
    #     img_array = transform(image=img_array)['image']

    # elif aug_type == 'noise':
    #     # Keep noise incredibly subtle so it doesn't skew pixel means dramatically
    #     transform = A.GaussNoise(std_range=(0.01, 0.04), p=1.0)
    #     img_array = transform(image=img_array)['image']

    # elif aug_type == 'blur':
    #     # Simulates a slightly shaky hand/phone camera without changing color values
    #     transform = A.GaussianBlur(blur_limit=(3, 5), p=1.0)
    #     img_array = transform(image=img_array)['image']

    # elif aug_type == 'horizontal_flip':
    #     transform = A.HorizontalFlip(p=1.0)
    #     img_array = transform(image=img_array)['image']

    # # 5. Convert back to PIL Image and encode back to base64 string
    # img = Image.fromarray(img_array)
    # buffer = io.BytesIO()
    # img.save(buffer, format='JPEG', quality=95)
    # augmented_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    #PREVIOUS:

    img_array = np.array(img, dtype=np.float32)

    # Choose random augmentation type
    aug_type = augmentation_type or random.choice([
        'rotation', 'brightness', 'contrast', 'color_shift', 'noise', 'gamma'
    ])

    # Apply augmentation
    if aug_type == 'rotation':
        angle = random.uniform(-15, 15)
        img = Image.fromarray(np.uint8(img_array)).rotate(angle, expand=False)
        img_array = np.array(img, dtype=np.float32)
    elif aug_type == 'brightness':
        delta = random.uniform(-18, 18)
        img_array = np.clip(img_array + delta, 0, 255)
    elif aug_type == 'contrast':
        factor = random.uniform(0.85, 1.15)
        img = Image.fromarray(np.uint8(img_array))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        img_array = np.array(img, dtype=np.float32)
    elif aug_type == 'color_shift':
        shift_channel = random.choice([0, 1, 2])
        delta = random.uniform(-30, 30)
        img_array[:, :, shift_channel] = np.clip(
            img_array[:, :, shift_channel] + delta, 0, 255
        )
    elif aug_type == 'noise':
        if random.random() < 0.5:
            noise = np.random.normal(0, 2, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
    elif aug_type == 'gamma':
        if random.random() < 0.5:
            gamma = random.uniform(0.9, 1.1)
            img_array = 255.0 * np.power(img_array / 255.0, gamma)
            img_array = np.clip(img_array, 0, 255)

    # Convert back to image and encode as base64
    img_array = np.uint8(np.clip(img_array, 0, 255))
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    augmented_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
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
