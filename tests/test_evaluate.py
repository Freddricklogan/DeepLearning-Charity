import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from charity_model.evaluate import metrics, reliability, roc_points, threshold_sweep


def test_metrics_match_sklearn_and_confusion_counts() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(np.int64)
    prob = np.clip(y * 0.5 + rng.normal(0.3, 0.3, 500), 0, 1)
    m = metrics(y, prob, 0.5)
    pred = (prob >= 0.5).astype(int)
    assert m.accuracy == pytest.approx(accuracy_score(y, pred))
    assert m.precision == pytest.approx(precision_score(y, pred))
    assert m.recall == pytest.approx(recall_score(y, pred))
    assert m.f1 == pytest.approx(f1_score(y, pred))
    assert m.roc_auc == pytest.approx(roc_auc_score(y, prob))
    assert m.tp + m.fp + m.tn + m.fn == 500
    assert m.base_rate == pytest.approx(y.mean())
    assert m.as_dict()["n"] == 500


def test_metrics_edge_cases() -> None:
    y = np.array([1, 1, 1, 1], dtype=np.int64)
    prob = np.array([0.1, 0.2, 0.3, 0.4])
    m = metrics(y, prob)
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0
    assert np.isnan(m.roc_auc)  # one class only
    with pytest.raises(ValueError, match="same shape"):
        metrics(np.array([1], dtype=np.int64), np.array([0.5, 0.5]))


def test_reliability_and_sweep_and_roc() -> None:
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.int64)
    prob = np.array([0.05, 0.15, 0.95, 0.85, 0.55, 0.45, 0.35, 0.65])
    table = reliability(y, prob, bins=10)
    assert sum(n for _, _, n in table) == 8
    assert all(0 <= p <= 1 and 0 <= o <= 1 for p, o, _ in table)
    top = next(row for row in table if row[0] >= 0.9)
    assert top[1] == 1.0  # the top bin holds only the 0.95 row, a positive
    sweep = threshold_sweep(y, prob, steps=19)
    assert len(sweep) == 19
    recalls = [m.recall for m in sweep]
    assert recalls == sorted(recalls, reverse=True)  # recall is non-increasing in the threshold
    pts = roc_points(y, prob, steps=10)
    assert pts[0] == (0.0, 0.0) and pts[-1] == (1.0, 1.0)
    fprs = [f for f, _ in pts]
    assert fprs == sorted(fprs)
