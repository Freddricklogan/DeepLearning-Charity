from pathlib import Path

import numpy as np
import pytest

from charity_model.baseline import fit_predict
from charity_model.data import load, preprocess, split_scale
from charity_model.evaluate import metrics
from charity_model.network import NetConfig, train_predict

DATA = Path(__file__).resolve().parents[1] / "data" / "charity_data.csv.gz"


@pytest.fixture(scope="module")
def split():  # type: ignore[no-untyped-def]
    return split_scale(preprocess(load(DATA).head(3000)), seed=3)


def test_baseline_probabilities_and_better_than_base_rate(split) -> None:  # type: ignore[no-untyped-def]
    prob = fit_predict(split.x_train, split.y_train, split.x_test, seed=3)
    assert prob.shape == split.y_test.shape
    assert prob.min() >= 0 and prob.max() <= 1
    m = metrics(split.y_test, prob)
    assert m.accuracy > max(m.base_rate, 1 - m.base_rate) - 0.02  # at least about the majority rate


def test_network_trains_deterministically_on_a_small_slice(split) -> None:  # type: ignore[no-untyped-def]
    cfg = NetConfig(hidden=(16,), max_epochs=3, patience=2, batch_size=128)
    a = train_predict(split.x_train, split.y_train, split.x_test, split.y_test, cfg, seed=7)
    b = train_predict(split.x_train, split.y_train, split.x_test, split.y_test, cfg, seed=7)
    assert a.prob.shape == split.y_test.shape
    assert a.prob.min() >= 0 and a.prob.max() <= 1
    assert 1 <= a.epochs_run <= 3 and 1 <= a.best_epoch <= a.epochs_run
    assert set(a.history) >= {"loss", "val_loss", "accuracy", "val_accuracy"}
    # dense (in+1)*16 + batchnorm 4 per unit + output 16+1
    assert a.params == (split.x_train.shape[1] + 1) * 16 + 4 * 16 + 17
    assert np.allclose(a.prob, b.prob, atol=1e-5)
