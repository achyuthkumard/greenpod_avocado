# Avocado Ripening Stage Classification

## Greenpod Labs — Data Science Assignment 2

## 1. Objective

The objective is to develop and compare two computer vision approaches
for classifying avocado ripening stages from images.

## 2. Dataset Understanding

- Dataset source: Google Drive link provided in the assignment
- Total images: [Enter actual count]
- Ripening classes: [Enter actual class names]
- Valid images: [Enter count]
- Corrupted images: [Enter count]

### Class Distribution

[Insert class counts and outputs/class_distribution.png]

### Data Quality and Bias

Discuss:

- Class imbalance
- Corrupted or unreadable images
- Image dimensions and formats
- Exact duplicate images
- Lighting, background, camera, and variety differences
- Potential label errors

Only report issues that were actually checked or observed.

## 3. Data Preprocessing

- Verified image readability.
- Created a reproducible train/validation/test split.
- Training: approximately 60%
- Validation: approximately 20%
- Testing: approximately 20%
- Random seed: 42

Images that may depict the same physical avocado should be grouped
before splitting to avoid data leakage.

## 4. Approach 1: Classical Computer Vision + SVM

### Method

- Resize images to 128 × 128.
- Extract HSV colour histograms.
- Extract Local Binary Pattern (LBP) texture features.
- Standardize features.
- Train an RBF-kernel Support Vector Machine.
- Use class weighting to help address class imbalance.

### Why This Approach?

It provides an interpretable, relatively lightweight baseline
without requiring a large neural network.

### Results

- Test accuracy: [Enter measured value]
- Macro precision: [Enter measured value]
- Macro recall: [Enter measured value]
- Macro F1-score: [Enter measured value]
- Mean inference time: [Enter measured value]

## 5. Approach 2: MobileNetV2 Transfer Learning

### Method

- Resize images to 160 × 160.
- Load ImageNet-pretrained MobileNetV2.
- Freeze the pretrained feature extractor.
- Add global average pooling, dropout, and a classification layer.
- Train with validation monitoring and early stopping.

### Why This Approach?

Transfer learning reuses visual features learned from a large image
dataset and can be effective when the target dataset is limited.

### Results

- Test accuracy: [Enter measured value]
- Macro precision: [Enter measured value]
- Macro recall: [Enter measured value]
- Macro F1-score: [Enter measured value]
- Mean inference time: [Enter measured value]

## 6. Model Comparison

Insert the contents of outputs/final_model_comparison.csv.

Compare:

- Accuracy
- Macro precision, recall, and F1-score
- Per-class performance
- Inference time
- Model size and deployment requirements

Choose the model using validation results. Use the test set for
the final evaluation of the selected model.

## 7. Example Prediction

Include:

- A sample avocado image
- Actual label, if known
- Classical SVM prediction
- MobileNetV2 prediction
- Confidence estimates, where available
- Whether the predictions agree

Explain any misclassification honestly.

## 8. Production Performance

The model can serve as an initial image-based ripeness classifier.
Real-world performance must be validated under different lighting,
camera settings, backgrounds, avocado varieties, and image quality.

Measure end-to-end latency, throughput, memory usage, and per-class
recall before production deployment.

## 9. Limitations

- Colour changes do not always correspond consistently to ripeness.
- Results depend on dataset size and representativeness.
- Unseen varieties and environments can reduce performance.
- External images cannot reliably reveal internal fruit quality.
- Confidence scores are not necessarily calibrated probabilities.

## 10. Future Improvements

- Collect more diverse, expertly labelled images.
- Prevent leakage using group-based splitting where necessary.
- Tune hyperparameters using validation data.
- Evaluate segmentation and stronger data augmentation.
- Calibrate confidence and add an uncertain-prediction option.
- Benchmark on an independent production-like dataset.
- Optimize inference and deploy a monitored API.

## 11. Mandatory Questions

### Q1. How will the model perform in production?

[Discuss measured performance and the additional real-world testing needed.]

### Q2. What are the drawbacks of your model?

[Discuss the observed weaknesses of both approaches.]

### Q3. What would you improve with more time?

[Describe realistic improvements based on your findings.]

### Q4. Where did you spend the most time?

[Describe your actual time spent on dataset inspection, preprocessing,
feature extraction, training, debugging, and evaluation.]

## 12. Conclusion

Summarize the measured results of both approaches and justify the
final model selection using evaluation metrics and deployment constraints.
