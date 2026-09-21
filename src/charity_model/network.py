"""Keras 3 multilayer perceptron on the JAX backend. Architecture and training are explicit
arguments so the report can print exactly what produced its numbers."""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NetConfig:
    hidden: tuple[int, ...] = (80, 30)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5


@dataclass(frozen=True)
class Trained:
    prob: NDArray[np.float64]
    epochs_run: int
    best_epoch: int
    history: dict[str, list[float]]
    params: int


def build(input_dim: int, cfg: NetConfig, seed: int) -> keras.Model:
    keras.utils.set_random_seed(seed)
    layers: list[keras.layers.Layer] = [keras.Input(shape=(input_dim,))]
    for units in cfg.hidden:
        layers += [
            keras.layers.Dense(units, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(cfg.dropout),
        ]
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_predict(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
    cfg: NetConfig,
    seed: int = 42,
) -> Trained:
    """Train with early stopping on validation loss (restoring the best weights) and predict."""
    model = build(x_train.shape[1], cfg, seed)
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience, restore_best_weights=True
    )
    # Labels go in as float32: JAX's default int32 would otherwise truncate int64 with a warning.
    hist = model.fit(
        x_train,
        y_train.astype(np.float32),
        validation_data=(x_test, y_test.astype(np.float32)),
        epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        callbacks=[stop],
        verbose=0,
    )
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    epochs_run = len(history["loss"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    prob = np.asarray(model.predict(x_test, verbose=0), dtype=np.float64).reshape(-1)
    return Trained(
        prob=prob,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        history=history,
        params=int(model.count_params()),
    )
