import numpy as np

from src.data_pipeline import normalize_images, one_hot_encode, preprocess_external_image, reshape_images


def test_normalize_images_range() -> None:
    arr = np.array([[0, 128, 255]], dtype=np.uint8)
    out = normalize_images(arr)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_reshape_images_shape() -> None:
    arr = np.zeros((4, 28, 28), dtype=np.float32)
    out = reshape_images(arr)
    assert out.shape == (4, 28, 28, 1)


def test_one_hot_encode_shape() -> None:
    labels = np.array([0, 1, 9])
    encoded = one_hot_encode(labels)
    assert encoded.shape == (3, 10)
    assert float(encoded[0, 0]) == 1.0


def test_preprocess_external_image_shape_and_range() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:44, 20:44] = 255
    out = preprocess_external_image(image)
    assert out.shape == (1, 28, 28, 1)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_preprocess_external_image_white_background_dark_digit() -> None:
    image = np.full((80, 80), 255, dtype=np.uint8)
    image[20:60, 28:52] = 0
    out = preprocess_external_image(image)[0, :, :, 0]
    center_mean = float(out[10:18, 10:18].mean())
    corner_mean = float(out[0:4, 0:4].mean())
    assert center_mean > corner_mean


def test_preprocess_external_image_black_background_bright_digit() -> None:
    image = np.zeros((80, 80), dtype=np.uint8)
    image[20:60, 28:52] = 255
    out = preprocess_external_image(image)[0, :, :, 0]
    center_mean = float(out[10:18, 10:18].mean())
    corner_mean = float(out[0:4, 0:4].mean())
    assert center_mean > corner_mean
