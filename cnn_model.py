
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
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

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"{SPLIT_FILE} not found. Complete Step 3 first."
    )

df = pd.read_csv(SPLIT_FILE)

required = {"image_path", "label", "split"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required}")

for split_name in ["train", "validation", "test"]:
    if df[df["split"] == split_name].empty:
        raise ValueError(f"No images found in {split_name} split.")

# Consistent label mapping across all splits.
encoder = LabelEncoder()
encoder.fit(df["label"].astype(str))
class_names = list(encoder.classes_)
num_classes = len(class_names)

if num_classes < 2:
    raise ValueError("At least two classes are required.")

label_map = {name: i for i, name in enumerate(class_names)}

def make_dataset(frame, training=False):
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["label"].astype(str).map(label_map).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(
            buffer_size=len(frame), seed=SEED,
            reshuffle_each_iteration=True
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# MobileNetV2 pretrained on ImageNet; initially freeze its weights.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(
    num_classes, activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint_path = MODEL_DIR / "mobilenetv2_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
]

print("Classes:", class_names)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

start = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

training_seconds = time.perf_counter() - start

# Evaluate the best validation-selected model.
best_model = tf.keras.models.load_model(checkpoint_path)

probabilities = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)
y_true = test_df["label"].astype(str).map(label_map).to_numpy()

test_accuracy = accuracy_score(y_true, y_pred)
test_macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n--- Test Results: MobileNetV2 ---")
print("Accuracy:", round(test_accuracy, 4))
print("Macro F1:", round(test_macro_f1, 4))
print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    zero_division=0,
))
print("Training time (seconds):", round(training_seconds, 2))

cm = confusion_matrix(
    y_true, y_pred, labels=np.arange(num_classes)
)

pd.DataFrame(
    cm, index=class_names, columns=class_names
).to_csv(OUTPUT_DIR / "mobilenet_confusion_matrix.csv")

# Save label mapping for future inference.
with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

pd.DataFrame([{
    "model": "MobileNetV2 Transfer Learning",
    "test_accuracy": test_accuracy,
    "test_macro_f1": test_macro_f1,
    "training_seconds": training_seconds,
}]).to_csv(OUTPUT_DIR / "mobilenet_results.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("MobileNetV2 Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mobilenet_training_history.png", dpi=150)
plt.show()

print("\nSaved model:", checkpoint_path)
print("Saved metrics and confusion matrix in outputs/.")
