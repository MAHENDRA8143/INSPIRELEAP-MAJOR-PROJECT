from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Ensure src module is accessible from project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_pipeline import preprocess_external_image
from src.explainability import compute_gradcam, overlay_gradcam_on_digit

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:  # pragma: no cover
    st_canvas = None


MODEL_PATH = Path("models") / "cnn_model.h5"


@st.cache_resource
def load_model() -> tf.keras.Model | None:
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)
    return None


def infer(model: tf.keras.Model, image: np.ndarray) -> tuple[int, float, np.ndarray]:
    model_input = preprocess_external_image(image)
    probs = model.predict(model_input, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs


st.set_page_config(page_title="Advanced Digit Recognition", page_icon="AI", layout="wide")
st.title("Advanced Handwritten Digit Recognition Platform")
st.caption("Enterprise-grade digit AI system with explainability")

model = load_model()
if model is None:
    st.warning("No trained model found at models/cnn_model.h5. Run training first.")
    st.stop()

left, right = st.columns(2)
image_to_process: np.ndarray | None = None

with left:
    st.subheader("Upload Digit Image")
    uploaded_file = st.file_uploader("Drag and drop a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=240)

    st.subheader("Or Draw Digit")
    if st_canvas is not None:
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        if canvas.image_data is not None:
            canvas_img = canvas.image_data.astype(np.uint8)
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_RGBA2GRAY)
            image_to_process = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        st.info("Install streamlit-drawable-canvas for draw mode support.")

with right:
    st.subheader("Prediction")
    if image_to_process is not None:
        pred, conf, probs = infer(model, image_to_process)
        st.metric(label="Predicted Digit", value=str(pred), delta=f"{conf * 100:.2f}% confidence")

        probs_df = pd.DataFrame({"digit": list(range(10)), "probability": probs})
        st.bar_chart(probs_df.set_index("digit"))

        show_cam = st.toggle("Show Grad-CAM", value=True)
        if show_cam:
            input_img = preprocess_external_image(image_to_process)
            heatmap = compute_gradcam(model, input_img)
            overlay = overlay_gradcam_on_digit(input_img.squeeze(), heatmap)
            st.image(overlay, caption="Grad-CAM Overlay", channels="BGR", width=280)
    else:
        st.info("Upload an image or draw a digit to get predictions.")
