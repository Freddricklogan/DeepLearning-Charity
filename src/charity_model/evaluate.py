"""Classification metrics from labels and probabilities, computed here so the report cannot quote a
number the code did not produce."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Metrics:
    n: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tp: int
    fp: int
    tn: int
    fn: int
    base_rate: float  # share of positives; the accuracy of always predicting the majority class

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def metrics(y: NDArray[np.int64], prob: NDArray[np.float64], threshold: float = 0.5) -> Metrics:
    if y.shape != prob.shape or y.size == 0:
        msg = "y and prob must be non-empty and the same shape"
        raise ValueError(msg)
    pred = (prob >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    return Metrics(
        n=int(y.size),
        threshold=threshold,
        accuracy=(tp + tn) / y.size,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=auc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        base_rate=float(np.mean(y)),
    )


def reliability(
    y: NDArray[np.int64], prob: NDArray[np.float64], bins: int = 10
) -> list[tuple[float, float, int]]:
    """Calibration table: (mean predicted probability, observed rate, count) per equal-width bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[tuple[float, float, int]] = []
    for lo, hi in pairwise(edges):
        mask = (prob >= lo) & ((prob < hi) if hi < 1.0 else (prob <= hi))
        if mask.any():
            out.append((float(prob[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return out


def threshold_sweep(
    y: NDArray[np.int64], prob: NDArray[np.float64], steps: int = 19
) -> list[Metrics]:
    """Metrics at thresholds 0.05 … 0.95, for choosing an operating point deliberately."""
    return [metrics(y, prob, float(t)) for t in np.linspace(0.05, 0.95, steps)]


def roc_points(
    y: NDArray[np.int64], prob: NDArray[np.float64], steps: int = 50
) -> list[tuple[float, float]]:
    """(false-positive rate, true-positive rate) at evenly spaced thresholds, for the curve."""
    pts: list[tuple[float, float]] = []
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    for t in np.linspace(1.0, 0.0, steps + 1):
        pred = prob >= t
        pts.append((float(np.sum(pred & (y == 0)) / neg), float(np.sum(pred & (y == 1)) / pos)))
    return pts
