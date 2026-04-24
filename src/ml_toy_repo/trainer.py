# src/ml_toy_repo/trainer.py
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml_toy_repo.state import JobState, JobStatus, TrainingMetrics

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "mnist.h5"
EPOCHS = 3


def get_device_name() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return gpus[0].name
    return tf.config.list_physical_devices("CPU")[0].name


def load_data():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train = x_train[..., np.newaxis]
    return x_train, y_train


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_training(state: JobState) -> None:
    try:
        state.status = JobStatus.RUNNING
        state.metrics.device = get_device_name()
        state.metrics.start_time = time.time()
        state.metrics.epochs = EPOCHS

        x_train, y_train = load_data()
        model = build_model()
        history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=1)

        MODELS_DIR.mkdir(exist_ok=True)
        model.save(MODEL_PATH, save_format="h5")

        state.metrics.end_time = time.time()
        state.metrics.duration_seconds = round(
            state.metrics.end_time - state.metrics.start_time, 2
        )
        state.metrics.final_accuracy = round(
            float(history.history["accuracy"][-1]), 4
        )
        state.status = JobStatus.COMPLETE

    except Exception as e:
        state.status = JobStatus.FAILED
        state.error = str(e)