from pathlib import Path
import os
import subprocess

PATH = Path('/Users/vineshuniyal/Downloads/0.00')

for dng in PATH.rglob("*.dng"):
  subprocess.run([
    "exiftool",
    "-overwrite_original", 
    "-n", 
    "-Orientation=8", 
    str(dng)
  ], check = True)

print("Done!")