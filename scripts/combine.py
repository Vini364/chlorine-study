from pathlib import Path
import shutil

src = Path("/Users/vineshuniyal/Desktop/chlorine-study/dataset/Dataset_3")
dst = Path("/Users/vineshuniyal/Desktop/chlorine-study/dataset/Dataset_5")

for file in src.rglob("*"):
  if file.is_file():
    target = dst / file.relative_to(src)
    target.parent.mkdir(parents = True, exist_ok = True)
    shutil.copy2(file, target)