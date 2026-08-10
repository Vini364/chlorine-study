#!/usr/bin/env python3
"""
Convert folder-organized image dataset to JSON format.

Folder structure:
  data_folder/
    ├── 0.5/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── 1.0/
    │   ├── image1.jpg
    │   └── ...
    └── 2.0/
        └── ...

Output: JSON file with base64-encoded images and concentration labels
"""

import os
import json
import base64
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from scripts.device_tag import model_from_exif

# Register HEIC format with PIL
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed. HEIC files will not be supported.")
    print("Install with: pip install pillow-heif")

# Supported image formats (including HEIC from Apple devices)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic', '.heif', '.dng', '.tiff'}

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory"""
    images = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            images.append(file)
    return sorted(images)


def convert_folders_to_json(root_dir: str, output_file: str = "training/new_dataset.json") -> Tuple[bool, str]:
    """
    Convert folder-structured dataset to JSON format

    Args:
        root_dir: Path to root folder containing concentration subfolders
        output_file: Path to output JSON file

    Returns:
        Tuple of (success: bool, message: str)
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        return False, f"Root directory not found: {root_dir}"

    if not root_path.is_dir():
        return False, f"Path is not a directory: {root_dir}"

    dataset = []
    total_images = 0
    concentration_counts = {}

    print(f"Scanning directory: {root_dir}")
    print("=" * 70)

    # Iterate through subdirectories (each is a concentration level)
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    if not subdirs:
        return False, "No subdirectories found. Expected folders named by concentration values."

    for subdir in subdirs:
        folder_name = subdir.name

        # Try to parse concentration from folder name
        try:
            concentration = float(folder_name)
        except ValueError:
            print(f"⚠ Skipping '{folder_name}' - not a valid concentration value")
            continue

        # Get all images in this concentration folder
        images = get_image_files(subdir)

        if not images:
            print(f"⚠ No images found in {folder_name}/")
            continue

        print(f"\n✓ Processing concentration {concentration} mg/L:")
        print(f"  Found {len(images)} images")

        # Process each image
        for image_path in images:
            try:
                # base64_data = encode_image_to_base64(image_path)
                dataset.append({
                    "image": str(image_path.resolve()), # Store the absolute path instead of base64 for now
                    "concentration": concentration,
                    "filename": image_path.name,
                    "model": model_from_exif(str(image_path.resolve()))
                })
                total_images += 1
            except Exception as e:
                print(f"  ✗ Error processing {image_path.name}: {e}")
                continue

        concentration_counts[concentration] = len(images)

    if not dataset:
        return False, "No valid images found in any concentration folder"

    # Save to JSON
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(dataset, f)

        print("\n" + "=" * 70)
        print(f"✓ Successfully converted dataset!")
        print(f"  Total images: {total_images}")
        print(f"  Concentration levels: {len(concentration_counts)}")
        print(f"  Saved to: {output_file}")
        print(f"\nConcentration breakdown:")
        for conc in sorted(concentration_counts.keys()):
            print(f"  {conc} mg/L: {concentration_counts[conc]} images")

        return True, "Conversion successful"

    except Exception as e:
        return False, f"Error saving JSON file: {e}"

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python convert_images_to_json.py <path_to_data_folder> [output_file.json]")
        print("\nExample:")
        print("  python convert_images_to_json.py ~/Downloads/chlorine_data training/new_dataset.json")
        sys.exit(1)

    data_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "training/new_dataset.json"

    print("\nChlorine Dataset Conversion Tool")
    print("=" * 70)
    print(f"Input folder: {data_path}")
    print(f"Output file: {output_file}\n")

    success, message = convert_folders_to_json(data_path, output_file)

    if success:
        print(f"\n{message}")
        print("\nNext steps:")
        print("  1. Run the training pipeline:")
        print("     python -m training.pipeline")
        print("  2. The pipeline will:")
        print("     - Augment the images")
        print("     - Extract 36 features per image")
        print("     - Train Linear and Ensemble models")
        print("     - Save model weights to models/saved_artifacts/")
    else:
        print(f"\n✗ Conversion failed: {message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
