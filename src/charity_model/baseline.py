"""Logistic-regression baseline. A network that cannot beat this has not earned its complexity."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


def fit_predict(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float32],
    seed: int = 42,
) -> NDArray[np.float64]:
    """Fit L2-regularised logistic regression and return positive-class probabilities on x_test."""
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(x_train, y_train)
    prob: NDArray[np.float64] = model.predict_proba(x_test)[:, 1].astype(np.float64)
    return prob
