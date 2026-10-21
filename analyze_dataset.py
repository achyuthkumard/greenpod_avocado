
from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

DATASET_DIR = Path(r"E:\Greenpod_avacado_ass\dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.exists():
    print(f"Dataset folder not found: {DATASET_DIR}")
    print("Update DATASET_DIR to the actual extracted dataset folder.")
    raise SystemExit(1)

image_paths = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total image files found: {len(image_paths)}")

if not image_paths:
    print("No images found. Check the dataset folder.")
    raise SystemExit(1)

records = []
corrupted = []
hashes = {}

for path in image_paths:
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size
            image_format = img.format

        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(file_hash, []).append(str(path))

        records.append({
            "image_path": str(path),
            "class": path.parent.name,
            "width": width,
            "height": height,
            "format": image_format,
            "aspect_ratio": round(width / height, 3)
        })

    except Exception as error:
        corrupted.append({
            "image_path": str(path),
            "error": str(error)
        })

df = pd.DataFrame(records)

print(f"\nValid images: {len(df)}")
print(f"Corrupted images: {len(corrupted)}")

print("\nClass distribution:")
class_counts = df["class"].value_counts().sort_index()
print(class_counts.to_string())

if len(class_counts) > 1 and class_counts.min() > 0:
    ratio = class_counts.max() / class_counts.min()
    print(f"\nLargest/smallest class ratio: {ratio:.2f}")

print("\nImage dimensions:")
print(df[["width", "height"]].describe())

duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
print(f"\nExact duplicate groups: {len(duplicate_groups)}")

df.to_csv("dataset_quality.csv", index=False)

if corrupted:
    pd.DataFrame(corrupted).to_csv("corrupted_images.csv", index=False)

plt.figure(figsize=(9, 5))
class_counts.plot(kind="bar")
plt.title("Avocado Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Image Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

print("\nAnalysis completed.")
print("Generated: dataset_quality.csv and class_distribution.png")
