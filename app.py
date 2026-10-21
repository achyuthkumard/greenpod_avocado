
from pathlib import Path
import json
import time

import cv2
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from skimage.feature import local_binary_pattern


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / "models"

SVM_PATH = MODEL_DIR / "classical_svm.joblib"
CNN_PATH = MODEL_DIR / "mobilenetv2_best.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"

st.set_page_config(
    page_title="Avocado AI Studio",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PREMIUM DARK GREEN DESIGN
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #101713;
    color: #edf4ee;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(16, 23, 19, 0.92);
}

[data-testid="stSidebar"] {
    background: #151f19;
    border-right: 1px solid #29372d;
}

[data-testid="stSidebar"] * {
    color: #e2ebe3;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f0f7f0 !important;
}

.hero {
    padding: 28px 30px;
    border-radius: 20px;
    background: linear-gradient(120deg, #20392a, #17251c 75%);
    border: 1px solid #385440;
    margin-bottom: 24px;
}

.eyebrow {
    color: #b5f28c;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(27px, 4vw, 42px);
    line-height: 1.15;
    font-weight: 700;
    color: #f1f8f0;
    margin: 12px 0;
}

.hero-subtitle {
    color: #b6c8b9;
    font-size: 15px;
    max-width: 720px;
    line-height: 1.7;
}

.metric-card {
    background: #19231c;
    border: 1px solid #304034;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    font-size: 12px;
    color: #9eb2a1;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #d7f7c7;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    margin-top: 8px;
}

.panel {
    background: #17211a;
    border: 1px solid #2c3d30;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
}

.panel-caption {
    color: #9eb2a1;
    font-size: 13px;
    line-height: 1.6;
}

.prediction {
    color: #d7f7c7;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin: 8px 0;
    overflow-wrap: anywhere;
}

.status-pill {
    display: inline-block;
    color: #c8f4b7;
    background: #243d2a;
    border: 1px solid #456d48;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .6px;
}

div.stButton > button {
    background: #b5f28c;
    color: #142219;
    border: none;
    border-radius: 10px;
    min-height: 47px;
    font-weight: 700;
    width: 100%;
    transition: all .2s ease;
}

div.stButton > button:hover {
    background: #d0ffb4;
    color: #142219;
    border: none;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: #18251c;
    border: 1px dashed #57745b;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stTabs"] button {
    color: #b8cbb9;
}

[data-testid="stMetric"] {
    background: #19231c;
    border: 1px solid #304034;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    color: #d7f7c7;
}

hr {
    border-color: #2b3b2e;
}

footer {
    visibility: hidden;
}

.small-note {
    color: #93a797;
    font-size: 12px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    svm_model = joblib.load(SVM_PATH)
    cnn_model = tf.keras.models.load_model(CNN_PATH)

    with open(LABELS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    return svm_model, cnn_model, class_names


# --------------------------------------------------
# FEATURE EXTRACTION FOR SVM
# Must match classical_model.py
# --------------------------------------------------
def extract_svm_features(image_rgb):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (128, 128))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    for channel, hist_range in [
        (0, [0, 180]),
        (1, [0, 256]),
        (2, [0, 256]),
    ]:
        hist = cv2.calcHist(
            [hsv], [channel], None, [16], hist_range
        )
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

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

    return np.asarray(features, dtype=np.float32).reshape(1, -1)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='eyebrow'>AVOCADO AI STUDIO</div>",
        unsafe_allow_html=True,
    )
    st.markdown("## Navigation")
    page = st.radio(
        "Select a section",
        ["Image Analyzer", "About the Models", "Project Overview"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Model Pipeline")
    st.markdown("**01**  Image Upload")
    st.markdown("**02**  Feature Extraction")
    st.markdown("**03**  Model Inference")
    st.markdown("**04**  Prediction Comparison")

    st.divider()

    st.markdown(
        "<p class='small-note'>Greenpod Labs · Data Science Assignment 2"
        "<br>Experimental computer vision prototype</p>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">COMPUTER VISION / AGRITECH</div>
    <div class="hero-title">Avocado Ripeness Intelligence</div>
    <div class="hero-subtitle">
        Analyze avocado images using two machine-learning approaches.
        Compare handcrafted colour and texture features with deep-learning
        representations from MobileNetV2.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CHECK MODEL FILES
# --------------------------------------------------
missing_files = [
    path.name for path in [SVM_PATH, CNN_PATH, LABELS_PATH]
    if not path.exists()
]

if missing_files:
    st.error(
        "Required model files are missing: "
        + ", ".join(missing_files)
        + ". Run the corresponding training scripts first."
    )
    st.stop()

try:
    svm_model, cnn_model, class_names = load_models()
except Exception as exc:
    st.error(f"Could not load the trained models: {exc}")
    st.stop()


# --------------------------------------------------
# IMAGE ANALYZER
# --------------------------------------------------
if page == "Image Analyzer":
    st.markdown("## Ripeness analysis")
    st.markdown(
        "<p class='panel-caption'>Upload a clear image of one avocado. "
        "Both models will analyze the same image.</p>",
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns([1.25, 0.75], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload avocado image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )

        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                image_rgb = np.array(pil_image)

                st.image(
                    pil_image,
                    caption="Image selected for analysis",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Unable to open this image: {exc}")
                st.stop()
        else:
            st.markdown("""
            <div class="panel" style="text-align:center; padding:55px 20px;">
                <div style="font-size:52px;">🥑</div>
                <h3>Ready for analysis</h3>
                <p class="panel-caption">
                    Upload an image to begin the prediction process.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with info_col:
        st.markdown("""
        <div class="panel">
            <div class="eyebrow">MODEL 01</div>
            <h3>Classical Vision</h3>
            <p class="panel-caption">
                HSV colour histograms and LBP texture features,
                classified using an RBF-kernel SVM.
            </p>
            <span class="status-pill">FEATURE-BASED ML</span>
        </div>
        <div class="panel">
            <div class="eyebrow">MODEL 02</div>
            <h3>Deep Learning</h3>
            <p class="panel-caption">
                ImageNet-pretrained MobileNetV2 with a custom
                classification head for your dataset classes.
            </p>
            <span class="status-pill">TRANSFER LEARNING</span>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("✦  ANALYZE RIPENING STAGE", type="primary"):
            try:
                with st.spinner("Running both models..."):
                    # Classical model
                    svm_features = extract_svm_features(image_rgb)

                    start = time.perf_counter()
                    svm_prediction = str(
                        svm_model.predict(svm_features)[0]
                    )
                    svm_elapsed = time.perf_counter() - start

                    svm_confidence = None
                    if hasattr(svm_model, "predict_proba"):
                        svm_probs = svm_model.predict_proba(
                            svm_features
                        )[0]
                        svm_confidence = float(np.max(svm_probs))

                    # Deep-learning model.
                    # The model contains preprocess_input internally,
                    # so pass resized RGB pixel values in the 0-255 range.
                    cnn_input = tf.image.resize(
                        image_rgb, (160, 160)
                    )
                    cnn_input = tf.cast(cnn_input, tf.float32)
                    cnn_input = tf.expand_dims(cnn_input, axis=0)

                    start = time.perf_counter()
                    cnn_probs = cnn_model.predict(
                        cnn_input, verbose=0
                    )[0]
                    cnn_elapsed = time.perf_counter() - start

                    cnn_index = int(np.argmax(cnn_probs))
                    cnn_prediction = str(class_names[cnn_index])
                    cnn_confidence = float(cnn_probs[cnn_index])

                st.divider()
                st.markdown("## Analysis results")

                res1, res2, res3 = st.columns(3)

                with res1:
                    st.markdown(
                        "<div class='metric-card'>"
                        "<div class='metric-label'>SVM latency</div>"
                        f"<div class='metric-value'>{svm_elapsed * 1000:.1f} ms</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                with res2:
                    st.markdown(
                        "<div class='metric-card'>"
                        "<div class='metric-label'>MobileNetV2 latency</div>"
                        f"<div class='metric-value'>{cnn_elapsed * 1000:.1f} ms</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                with res3:
                    agreement = (
                        "Agreement" if svm_prediction == cnn_prediction
                        else "Disagreement"
                    )
                    st.markdown(
                        "<div class='metric-card'>"
                        "<div class='metric-label'>Model comparison</div>"
                        f"<div class='metric-value' style='font-size:21px'>{agreement}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                left, right = st.columns(2, gap="large")

                with left:
                    st.markdown(
                        "<div class='panel'>"
                        "<div class='eyebrow'>APPROACH 01</div>"
                        "<h3>Classical SVM</h3>"
                        f"<div class='prediction'>{svm_prediction}</div>"
                        "<p class='panel-caption'>HSV + LBP features</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    if svm_confidence is not None:
                        st.write("SVM probability estimate")
                        st.progress(
                            max(0.0, min(1.0, svm_confidence))
                        )
                        st.caption(
                            f"Maximum estimated class probability: "
                            f"{svm_confidence:.1%}"
                        )

                with right:
                    st.markdown(
                        "<div class='panel'>"
                        "<div class='eyebrow'>APPROACH 02</div>"
                        "<h3>MobileNetV2</h3>"
                        f"<div class='prediction'>{cnn_prediction}</div>"
                        "<p class='panel-caption'>Transfer learning</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    st.write("CNN maximum class score")
                    st.progress(
                        max(0.0, min(1.0, cnn_confidence))
                    )
                    st.caption(
                        f"Maximum model score: {cnn_confidence:.1%}"
                    )

                with st.expander("View MobileNetV2 class scores"):
                    for i, class_name in enumerate(class_names):
                        st.write(
                            f"**{class_name}** — "
                            f"{float(cnn_probs[i]):.1%}"
                        )
                        st.progress(
                            max(0.0, min(1.0, float(cnn_probs[i])))
                        )

                if svm_prediction == cnn_prediction:
                    st.success(
                        "Both models predict the same ripening class."
                    )
                else:
                    st.warning(
                        "The models disagree. Review the image and "
                        "compare both models' validation performance."
                    )

                st.info(
                    "This is an experimental image classifier. Scores "
                    "are not guaranteed to be calibrated probabilities "
                    "of correctness. External appearance alone cannot "
                    "confirm internal fruit quality."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# ABOUT THE MODELS
# --------------------------------------------------
elif page == "About the Models":
    st.markdown("## Two approaches. One classification task.")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="panel">
            <div class="eyebrow">APPROACH 01</div>
            <h2>Classical Computer Vision</h2>
            <h3>HSV + LBP + SVM</h3>
            <p class="panel-caption">
                Extracts colour histograms and local texture patterns.
                An RBF-kernel SVM uses these features to classify the image.
            </p>
            <hr>
            <b>Strengths</b>
            <p class="panel-caption">
                Lightweight features, relatively simple pipeline,
                and a useful baseline for smaller datasets.
            </p>
            <b>Limitations</b>
            <p class="panel-caption">
                Handcrafted features may be sensitive to lighting,
                backgrounds, and differences between avocado varieties.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
            <div class="eyebrow">APPROACH 02</div>
            <h2>Deep Learning</h2>
            <h3>MobileNetV2</h3>
            <p class="panel-caption">
                Uses pretrained visual representations and a custom
                classification layer to predict your ripening classes.
            </p>
            <hr>
            <b>Strengths</b>
            <p class="panel-caption">
                Learns useful visual patterns beyond manually
                designed colour and texture features.
            </p>
            <b>Limitations</b>
            <p class="panel-caption">
                Requires more compute and can overfit or generalize
                poorly when the dataset is small or biased.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "The actual performance comparison should use the same held-out "
        "test set and report accuracy, macro F1, per-class recall, "
        "and measured inference latency."
    )


# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------
elif page == "Project Overview":
    st.markdown("## Project overview")

    st.markdown("""
    <div class="panel">
        <div class="eyebrow">PROJECT OBJECTIVE</div>
        <h2>Avocado Ripening Stage Classification</h2>
        <p class="panel-caption">
            This project compares a classical computer vision pipeline
            with transfer learning to classify avocado images.
            The goal is to assess model quality, interpret trade-offs,
            and understand practical deployment limitations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Approaches", "2")
    c2.metric("Task", "Image classification")
    c3.metric("Framework", "OpenCV + TensorFlow")

    st.markdown("### Pipeline")
    st.markdown("""
    1. Dataset inspection and quality checks
    2. Train/validation/test split
    3. HSV + LBP feature extraction and SVM
    4. MobileNetV2 transfer learning
    5. Evaluation using accuracy, precision, recall and macro F1
    6. Prediction demo and model comparison
    """)

    st.markdown("### Production considerations")
    st.markdown("""
    - Test with new images collected under real operating conditions.
    - Check for class imbalance and leakage between data splits.
    - Measure latency, memory usage and throughput on the target device.
    - Monitor errors across classes, lighting conditions and varieties.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.markdown(
    "<p class='small-note' style='text-align:center'>"
    "AVOCADO AI STUDIO · GREENPOD LABS ASSIGNMENT 2<br>"
    "Experimental research prototype — not a food-safety assessment"
    "</p>",
    unsafe_allow_html=True,
)
