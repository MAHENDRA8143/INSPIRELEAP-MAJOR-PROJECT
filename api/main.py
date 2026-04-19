from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile

# Ensure src module is accessible from project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_pipeline import preprocess_external_image


APP = FastAPI(title="Advanced Handwritten Digit Recognition API", version="1.0.0")
MODEL_PATH = Path("models") / "cnn_model.h5"


def _load_model() -> tf.keras.Model:
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)
    # Fallback model keeps API health usable before training artifacts exist.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


MODEL = _load_model()


@APP.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(MODEL is not None)}


@APP.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    content = await file.read()
    np_buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    model_input = preprocess_external_image(image)
    probs = MODEL.predict(model_input, verbose=0)[0]
    pred_digit = int(np.argmax(probs))
    confidence = float(probs[pred_digit])

    sorted_probs = np.argsort(probs)[::-1]
    top_probs = [{"digit": int(i), "probability": float(probs[i])} for i in sorted_probs]

    return {
        "digit": pred_digit,
        "confidence": confidence,
        "probabilities": top_probs,
    }


app = APP
