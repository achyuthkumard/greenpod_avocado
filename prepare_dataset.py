
from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = Path(r"E:\Greenpod_avacado_ass\dataset")
OUTPUT_DIR = Path(r"E:\Greenpod_avacado_ass\outputs")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR}. Update DATASET_DIR."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

for path in DATASET_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:
        with Image.open(path) as img:
            img.verify()

        # Assumes images are stored in class-specific folders.
        records.append({
            "image_path": str(path.resolve()),
            "label": path.parent.name
        })

    except Exception:
        print(f"Skipping unreadable image: {path}")

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No valid images found.")

counts = df["label"].value_counts()

print("\nClass distribution:")
print(counts.to_string())

# Stratified splitting needs enough samples in every class.
if len(counts) < 2:
    raise ValueError(
        "At least two classes are needed for classification."
    )

if counts.min() < 3:
    raise ValueError(
        "At least one class has fewer than 3 valid images. "
        "A reliable stratified train/validation/test split "
        "is not possible yet. Inspect the dataset first."
    )

# First reserve 20% for testing.
train_val, test = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

# Use 20% of the remaining data for validation.
train, val = train_test_split(
    train_val,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=train_val["label"]
)

train["split"] = "train"
val["split"] = "validation"
test["split"] = "test"

splits = pd.concat([train, val, test], ignore_index=True)

splits.to_csv(OUTPUT_DIR / "dataset_splits.csv", index=False)

print("\nSplit summary:")
print(pd.crosstab(splits["label"], splits["split"]))

print("\nTotal images:", len(splits))
print("Training images:", len(train))
print("Validation images:", len(val))
print("Testing images:", len(test))
print("\nSaved:", OUTPUT_DIR / "dataset_splits.csv")
