from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def compute_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    target_layer_name: str = "last_conv",
    class_idx: int | None = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(target_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]).numpy())
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam_on_digit(image_28x28: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_28x28.squeeze()
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (28, 28))
    heatmap_u8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
    colored_heatmap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 1 - alpha, colored_heatmap, alpha, 0)
    return overlay
