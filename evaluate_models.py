
from pathlib import Path
import json
import time

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from skimage.feature import local_binary_pattern
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT = Path(r"E:\Greenpod_avacado_ass")
OUT = PROJECT / "outputs"
MODELS = PROJECT / "models"

split_file = OUT / "dataset_splits.csv"
svm_file = MODELS / "classical_svm.joblib"
cnn_file = MODELS / "mobilenetv2_best.keras"
labels_file = MODELS / "class_names.json"

for file in [split_file, svm_file, cnn_file, labels_file]:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file missing: {file}. "
            "Complete the earlier training steps first."
        )

df = pd.read_csv(split_file)
test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

with open(labels_file, encoding="utf-8") as f:
    class_names = json.load(f)

svm = joblib.load(svm_file)
cnn = tf.keras.models.load_model(cnn_file)

def extract_svm_features(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")

    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = []

    for channel, bins, limits in [
        (0, 16, [0, 180]),
        (1, 16, [0, 256]),
        (2, 16, [0, 256]),
    ]:
        hist = cv2.calcHist([hsv], [channel], None, [bins], limits)
        features.extend(cv2.normalize(hist, hist).flatten())

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 12),
        range=(0, 11), density=True
    )
    features.extend(lbp_hist)
    return np.asarray(features, dtype=np.float32).reshape(1, -1)

def load_cnn_image(image_path):
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (160, 160))
    return tf.cast(image, tf.float32)

true_labels = []
svm_predictions = []
cnn_predictions = []
svm_times = []
cnn_times = []

for _, row in test_df.iterrows():
    path = row["image_path"]
    true_label = str(row["label"])

    features = extract_svm_features(path)
    start = time.perf_counter()
    svm_prediction = svm.predict(features)[0]
    svm_times.append(time.perf_counter() - start)

    image_tensor = load_cnn_image(path)
    image_batch = tf.expand_dims(image_tensor, axis=0)

    start = time.perf_counter()
    probabilities = cnn.predict(image_batch, verbose=0)[0]
    cnn_times.append(time.perf_counter() - start)
    cnn_prediction = class_names[int(np.argmax(probabilities))]

    true_labels.append(true_label)
    svm_predictions.append(svm_prediction)
    cnn_predictions.append(cnn_prediction)

def report_model(name, predictions, times):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(
        true_labels, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        true_labels, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        true_labels, predictions, average="macro", zero_division=0
    )

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Mean prediction time/image: {np.mean(times) * 1000:.2f} ms")
    print(classification_report(
        true_labels, predictions,
        labels=class_names,
        zero_division=0
    ))

    cm = confusion_matrix(
        true_labels, predictions, labels=class_names
    )
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        OUT / f"{name.lower().replace(' ', '_')}_confusion.csv"
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "mean_prediction_ms": np.mean(times) * 1000,
    }

svm_results = report_model("Classical SVM", svm_predictions, svm_times)
cnn_results = report_model("MobileNetV2", cnn_predictions, cnn_times)

comparison = pd.DataFrame([svm_results, cnn_results])
comparison.to_csv(OUT / "final_model_comparison.csv", index=False)

errors = pd.DataFrame({
    "image_path": test_df["image_path"],
    "actual_label": true_labels,
    "svm_prediction": svm_predictions,
    "cnn_prediction": cnn_predictions,
})

errors["svm_correct"] = errors["actual_label"] == errors["svm_prediction"]
errors["cnn_correct"] = errors["actual_label"] == errors["cnn_prediction"]
errors.to_csv(OUT / "test_predictions.csv", index=False)

print("\n===== FINAL COMPARISON =====")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved final_model_comparison.csv and test_predictions.csv")
