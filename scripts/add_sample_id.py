from pathlib import Path
from scripts.image_utils import get_image_files
import subprocess
from scripts.device_tag import get_time_from_exif
import sys
from tqdm import tqdm
def add_metadata_to_image(root_dir):
    root_path = Path(root_dir)

    if not root_path.exists():
        return False, f"Root directory not found: {root_dir}"

    if not root_path.is_dir():
        return False, f"Path is not a directory: {root_dir}"
    
    print(f"Scanning directory: {root_dir}")
    print("=" * 70)

    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    if not subdirs:
        return False, "No subdirectories found. Expected folders named by concentration values."
    session_count = 1
    for subdir in tqdm(subdirs, desc="Processing folders"):
        folder_name = subdir.name
        images = get_image_files(subdir)

        if not images:
            print(f"⚠ No images found in {folder_name}/")
            continue
        for image_path in images:
            try:
                img_path = str(image_path.resolve())
                date_time = get_time_from_exif(img_path).split(" ")[0]
                date_time = date_time.replace(":","")
                subprocess.run([
                    "exiftool",
                    "-overwrite_original",
                    f"-UserComment={date_time}_{session_count}",
                    str(image_path)
                    ], 
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text = True
                )
                
            except Exception as e:
                print(f"  ✗ Error processing {image_path.name}: {e}")
                continue
        session_count+=1

    return True, "Adding source_id successful"

def source_id_extractor(image_path):
  if image_path.suffix.lower() not in {".jpg", ".jpeg", ".heic", ".png", ".dng"}:
    return "NOT A VALID IMAGE"

  try:
      result = subprocess.run(
          [
              "exiftool",
              "-s3",
              "-UserComment",
              str(image_path)
          ],
          capture_output=True,
          text=True,
          check=True
      )

      user_comment = result.stdout.strip()
      if not user_comment:
        raise ValueError(
            f"No UserComment/source_id found in {image_path.name}"
        )
      return user_comment
  except subprocess.CalledProcessError as e:
    print(f"✗ Error reading {image_path.name}: {e}")
    
def main():
    """Main entry point"""
    if len(sys.argv) < 1:
        print("Usage: python add_sample_id.py <path_to_dataset_folder>")
        print("\nExample:")
        print("  python add_sample_id.py ~/Downloads/chlorine_data")
        sys.exit(1)

    data_path = sys.argv[1]

    print("\nChlorine Dataset Source_id Addition Tool")
    print("=" * 70)
    print(f"Input folder: {data_path}")

    success, message = add_metadata_to_image(data_path)

    if success:
        print(f"\n{message}")
        print("\nNext steps:")
        print("  1. Convert images to json:")
        print("     python -m scripts.convert_images_to_json <path_to_data_folder> [output_file.json]")
        print("  2. The script will:")
        print("     Add Attributes including:")
        print("     - Image path")
        print("     - Concentration")
        print("     - filename")
        print("     - Phone Type(Model)")
        print("     - source_id")
    else:
        print(f"\n✗ Conversion failed: {message}")
        sys.exit(1)

if __name__ == "__main__":
    main()