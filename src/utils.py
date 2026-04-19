from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import tensorflow as tf


LOGGER_NAME = "digit_ai_system"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_directories() -> None:
    for path in ["data", "models", "plots", "logs", "mlruns"]:
        Path(path).mkdir(parents=True, exist_ok=True)


def register_model_version(model_path: Path, metrics: dict[str, Any]) -> None:
    registry_path = Path("models") / "model_registry.json"
    registry = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    entry = {
        "version": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_path": str(model_path),
        "metrics": metrics,
    }
    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def convert_to_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def convert_to_onnx(model: tf.keras.Model, output_path: Path, logger: logging.Logger) -> None:
    try:
        import tf2onnx
    except ImportError:
        logger.warning("tf2onnx not installed. Skipping ONNX export.")
        return

    try:
        _spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=_spec, opset=13)
        output_path.write_bytes(model_proto.SerializeToString())
        logger.info("ONNX model exported to %s", output_path)
    except Exception as exc:  # pragma: no cover
        logger.exception("ONNX conversion failed: %s", exc)


def export_optimized_models(model: tf.keras.Model, logger: logging.Logger) -> None:
    tflite_path = Path("models") / "model.tflite"
    onnx_path = Path("models") / "model.onnx"
    convert_to_tflite(model, tflite_path)
    logger.info("TFLite model exported to %s", tflite_path)
    convert_to_onnx(model, onnx_path, logger)
