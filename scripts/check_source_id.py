import subprocess
from pathlib import Path
import sys

def check_one_folder(root_dir):
  for image_path in root_dir.rglob("*"):
      if image_path.suffix.lower() not in {".jpg", ".jpeg", ".heic", ".png", ".dng"}:
          continue

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

          print(f"{image_path.name} -> {user_comment or 'NO UserComment'}")
          # return user_comment

      except subprocess.CalledProcessError as e:
          print(f"✗ Error reading {image_path.name}: {e}")

def check_root_folder(root_dir):
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
  for subdir in subdirs:
      print(f"For concentration folder: {subdir.name}")
      check_one_folder(subdir)
      print("="*50)

if __name__ == "__main__":
  root_dir = sys.argv[1]
  check_root_folder(root_dir)