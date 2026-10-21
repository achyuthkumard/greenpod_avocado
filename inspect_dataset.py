
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(r"E:\Greenpod_avacado_ass\dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.exists():
    print(f"Dataset folder not found: {DATASET_DIR}")
    raise SystemExit(1)

image_files = [
    path for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"\nTotal images found: {len(image_files)}")

if not image_files:
    print("No images found. Check the extracted folder path.")
    raise SystemExit(0)

# Count images by their immediate parent folder
class_counts = Counter(path.parent.name for path in image_files)

print("\nImage counts by folder:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\nSample image paths:")
for path in image_files[:10]:
    print(path)

print("\nTop-level dataset structure:")
for item in sorted(DATASET_DIR.iterdir()):
    if item.is_dir():
        print(f"[Folder] {item.name}")
    else:
        print(f"[File]   {item.name}")
