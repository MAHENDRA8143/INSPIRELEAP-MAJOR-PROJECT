from __future__ import annotations

from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MNIST_DATASET_PATH = DATA_DIR / "mnist.npz"
MNIST_DATASET_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def normalize_images(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def reshape_images(images: np.ndarray) -> np.ndarray:
    return images.reshape((-1, 28, 28, 1)).astype(np.float32)


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


def _elastic_distortion(image: np.ndarray, alpha: float = 8.0, sigma: float = 4.0) -> np.ndarray:
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return distorted


def _apply_contrast_normalization(image: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    normalized = cv2.equalizeHist(image_u8)
    return normalized.astype(np.float32) / 255.0


def advanced_preprocess_images(images: np.ndarray, apply_probability: float = 0.7) -> np.ndarray:
    processed = images.copy()
    for i in range(processed.shape[0]):
        if np.random.rand() > apply_probability:
            continue

        image = processed[i].squeeze()

        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.08, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if np.random.rand() < 0.4:
            image = _elastic_distortion(image)

        if np.random.rand() < 0.7:
            image = _apply_contrast_normalization(image)

        processed[i, :, :, 0] = np.clip(image, 0.0, 1.0)

    return processed


def get_data_generator() -> ImageDataGenerator:
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )


def load_mnist_dataset(
    enable_advanced_preprocessing: bool = True,
    path: str | Path = MNIST_DATASET_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        with np.load(dataset_path) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
    else:
        downloaded_path = Path(
            tf.keras.utils.get_file(
                fname=dataset_path.name,
                origin=MNIST_DATASET_URL,
                cache_dir=str(dataset_path.parent),
                cache_subdir="",
            )
        )
        if downloaded_path != dataset_path and downloaded_path.exists():
            dataset_path.write_bytes(downloaded_path.read_bytes())

        with np.load(dataset_path) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    x_train = reshape_images(x_train)
    x_test = reshape_images(x_test)

    if enable_advanced_preprocessing:
        x_train = advanced_preprocess_images(x_train)

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    return x_train, y_train, x_test, y_test


def preprocess_external_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Evaluate both polarity hypotheses and keep the one with digit-like area.
    _, mask_bright = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_dark = cv2.bitwise_not(mask_bright)

    bright_ratio = float(np.count_nonzero(mask_bright)) / float(mask_bright.size)
    dark_ratio = float(np.count_nonzero(mask_dark)) / float(mask_dark.size)

    target_ratio = 0.18
    use_inverted = abs(dark_ratio - target_ratio) < abs(bright_ratio - target_ratio)

    working = cv2.bitwise_not(image) if use_inverted else image
    mask = mask_dark if use_inverted else mask_bright

    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        resized = cv2.resize(working, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)
        return normalized.reshape(1, 28, 28, 1)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = working[y0:y1, x0:x1]

    h, w = crop.shape
    scale = 20.0 / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized.astype(np.float32)

    if canvas.max() > 0:
        canvas = canvas / 255.0

    canvas = np.clip(canvas, 0.0, 1.0)
    return canvas.reshape(1, 28, 28, 1)
