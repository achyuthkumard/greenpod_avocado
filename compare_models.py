
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(r"E:\Greenpod_avacado_ass")
OUTPUT_DIR = PROJECT_DIR / "outputs"

RESULT_FILES = [
    OUTPUT_DIR / "classical_results.csv",
    OUTPUT_DIR / "mobilenet_results.csv",
]

available_results = []

for file_path in RESULT_FILES:
    if file_path.exists():
        df = pd.read_csv(file_path)
        available_results.append(df)
        print(f"Loaded: {file_path.name}")
    else:
        print(f"Missing: {file_path.name}")

if len(available_results) != 2:
    print(
        "\nPlease run both classical_model.py and cnn_model.py "
        "successfully before comparing them."
    )
    raise SystemExit(1)

results = pd.concat(available_results, ignore_index=True)

required_columns = {
    "model", "test_accuracy", "test_macro_f1", "training_seconds"
}

if not required_columns.issubset(results.columns):
    raise ValueError(
        f"Result files must contain: {required_columns}"
    )

results = results.sort_values(
    "test_macro_f1", ascending=False
).reset_index(drop=True)

results.to_csv(
    OUTPUT_DIR / "model_comparison.csv", index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Plot accuracy
plt.figure(figsize=(9, 5))
plt.bar(results["model"], results["test_accuracy"])
plt.title("Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
plt.show()

# Plot macro F1-score
plt.figure(figsize=(9, 5))
plt.bar(results["model"], results["test_macro_f1"])
plt.title("Test Macro F1-Score Comparison")
plt.ylabel("Macro F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_comparison.png", dpi=150)
plt.show()

best_model = results.iloc[0]

print("\nBest model by test macro F1-score:")
print(best_model["model"])
print(f"Macro F1-score: {best_model['test_macro_f1']:.4f}")

print("\nSaved model_comparison.csv and comparison charts.")
