from pathlib import Path
from typing import List

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.heic', '.heif', '.dng', '.tiff'}

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory"""
    images = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            images.append(file)
    return sorted(images)

