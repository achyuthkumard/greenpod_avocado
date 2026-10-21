
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
import pandas as pd

from skimage.feature import local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(r"E:\Greenpod_avacado_ass")
SPLIT_FILE = PROJECT_DIR / "outputs" / "dataset_splits.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)


def extract_features(image_path):
    """Extract HSV colour histograms and LBP texture features."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # Colour histograms: 16 bins for each HSV channel.
    for channel in range(3):
        hist = cv2.calcHist(
            [hsv], [channel], None, [16],
            [0, 256] if channel == 2 else
            ([0, 180] if channel == 0 else [0, 256])
        )
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # Texture features using Local Binary Patterns.
    lbp = local_binary_pattern(
        gray, P=8, R=1, method="uniform"
    )
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 12),
        range=(0, 11),
        density=True,
    )
    features.extend(lbp_hist)

    return np.asarray(features, dtype=np.float32)


if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"{SPLIT_FILE} not found. Complete Step 3 first."
    )

df = pd.read_csv(SPLIT_FILE)

required_columns = {"image_path", "label", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Split file must contain: {required_columns}"
    )

X_by_split = {}
y_by_split = {}

print("Extracting image features...")

for split_name in ["train", "validation", "test"]:
    part = df[df["split"] == split_name]

    features = []
    labels = []

    for _, row in part.iterrows():
        try:
            features.append(extract_features(row["image_path"]))
            labels.append(row["label"])
        except Exception as exc:
            print(f"Skipping {row['image_path']}: {exc}")

    if not features:
        raise ValueError(f"No readable images in {split_name} split.")

    X_by_split[split_name] = np.vstack(features)
    y_by_split[split_name] = np.asarray(labels)

print("Training feature shape:", X_by_split["train"].shape)

model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
)

start = time.perf_counter()
model.fit(X_by_split["train"], y_by_split["train"])
training_seconds = time.perf_counter() - start

# Use validation data for initial model selection.
val_pred = model.predict(X_by_split["validation"])

print("\n--- Validation Results ---")
print("Accuracy:", round(
    accuracy_score(y_by_split["validation"], val_pred), 4
))
print("Macro F1:", round(
    f1_score(
        y_by_split["validation"], val_pred,
        average="macro", zero_division=0
    ), 4
))
print(classification_report(
    y_by_split["validation"], val_pred, zero_division=0
))

# Evaluate the baseline on the held-out test set.
test_pred = model.predict(X_by_split["test"])

test_accuracy = accuracy_score(y_by_split["test"], test_pred)
test_f1 = f1_score(
    y_by_split["test"], test_pred,
    average="macro", zero_division=0
)

print("\n--- Test Results: Classical SVM ---")
print("Accuracy:", round(test_accuracy, 4))
print("Macro F1:", round(test_f1, 4))
print(classification_report(
    y_by_split["test"], test_pred, zero_division=0
))
print("Training time (seconds):", round(training_seconds, 2))

labels = sorted(np.unique(y_by_split["test"]))
cm = confusion_matrix(
    y_by_split["test"], test_pred, labels=labels
)

pd.DataFrame(
    cm, index=labels, columns=labels
).to_csv(OUTPUT_DIR / "svm_confusion_matrix.csv")

joblib.dump(model, MODEL_DIR / "classical_svm.joblib")

pd.DataFrame([{
    "model": "HSV + LBP + SVM",
    "test_accuracy": test_accuracy,
    "test_macro_f1": test_f1,
    "training_seconds": training_seconds,
}]).to_csv(
    OUTPUT_DIR / "classical_results.csv", index=False
)

print("\nSaved model: models/classical_svm.joblib")
print("Saved results: outputs/classical_results.csv")
print("Saved confusion matrix: outputs/svm_confusion_matrix.csv")
