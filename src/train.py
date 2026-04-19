from __future__ import annotations

import random
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from src.data_pipeline import get_data_generator, load_mnist_dataset
from src.evaluate import evaluate_model, generate_error_analysis
from src.model import build_advanced_cnn
from src.utils import configure_logging, ensure_directories, export_optimized_models, register_model_version


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train() -> None:
    ensure_directories()
    logger = configure_logging()
    set_seed(42)

    logger.info("Loading and preprocessing MNIST dataset")
    x_train, y_train, x_test, y_test = load_mnist_dataset(enable_advanced_preprocessing=True)

    logger.info("Building advanced CNN model")
    model = build_advanced_cnn()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint("models/cnn_model.h5", monitor="val_accuracy", save_best_only=True),
        TensorBoard(log_dir="logs/tensorboard"),
    ]

    datagen = get_data_generator()
    datagen.fit(x_train)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("advanced-digit-recognition")

    with mlflow.start_run(run_name="advanced_cnn_mnist"):
        mlflow.log_params(
            {
                "optimizer": "AdamW",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 128,
                "epochs": 30,
            }
        )

        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=128),
            epochs=30,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        best_model = tf.keras.models.load_model("models/cnn_model.h5")
        metrics = evaluate_model(best_model, x_test, y_test, plot_dir=Path("plots"))
        y_prob = best_model.predict(x_test, verbose=0)
        generate_error_analysis(x_test, y_test, y_prob, Path("plots"), top_n=20)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifacts("plots")
        mlflow.log_artifact("models/cnn_model.h5")

        export_optimized_models(best_model, logger)
        register_model_version(Path("models/cnn_model.h5"), metrics)

        logger.info("Training complete. Final metrics: %s", metrics)
        logger.info("Epochs run: %d", len(history.history.get("loss", [])))


if __name__ == "__main__":
    train()
