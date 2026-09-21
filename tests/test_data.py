from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from charity_model.data import CATEGORICAL, REQUIRED, bin_rare, load, preprocess, split_scale

DATA = Path(__file__).resolve().parents[1] / "data" / "charity_data.csv.gz"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load(DATA)


def test_load_shape_and_schema(df: pd.DataFrame) -> None:
    assert df.shape == (34299, 12)
    assert all(c in df.columns for c in REQUIRED)
    assert set(df["IS_SUCCESSFUL"].unique()) == {0, 1}


def test_load_rejects_bad_schema(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("EIN,NAME\n1,x\n")
    with pytest.raises(ValueError, match="missing columns"):
        load(p)
    q = tmp_path / "bad2.csv"
    q.write_text(",".join(REQUIRED) + "\n" + ",".join(["1"] * 11) + ",2\n")
    with pytest.raises(ValueError, match="must be 0/1"):
        load(q)


def test_bin_rare_folds_only_below_threshold() -> None:
    s = pd.Series(["a"] * 5 + ["b"] * 2 + ["c"])
    out, rare = bin_rare(s, 3)
    assert rare == ["b", "c"]
    assert out.tolist() == ["a"] * 5 + ["Other"] * 3
    same, none = bin_rare(s, 1)
    assert none == []
    assert same.tolist() == s.tolist()


def test_preprocess_bins_encodes_and_logs(df: pd.DataFrame) -> None:
    p = preprocess(df)
    assert p.x.shape[0] == 34299
    assert p.x.dtype == np.float32
    assert len(p.features) == p.x.shape[1]
    # Identifiers are gone; categoricals became one-hot columns; STATUS and ASK_AMT stay numeric.
    assert not any(f.startswith(("EIN", "NAME")) for f in p.features)
    for col in CATEGORICAL:
        assert any(f.startswith(col + "_") for f in p.features)
    assert "ASK_AMT" in p.features and "STATUS" in p.features
    # Binning happened for the two configured columns and the rare values are listed.
    assert "APPLICATION_TYPE" in p.binned and "CLASSIFICATION" in p.binned
    assert "APPLICATION_TYPE_Other" in p.features
    # The classic thresholds keep 9 application types (8 + Other), the exercise's known result.
    assert sum(f.startswith("APPLICATION_TYPE_") for f in p.features) == 9
    ask = p.x[:, p.features.index("ASK_AMT")]
    assert ask.max() < 25  # log1p(8.6e9) ≈ 22.9, not 8.6e9
    assert ask.min() >= np.log1p(5000) - 1e-3


def test_preprocess_custom_thresholds(df: pd.DataFrame) -> None:
    p = preprocess(df, {"APPLICATION_TYPE": 10**9})
    # Everything folded into Other: exactly one APPLICATION_TYPE column.
    assert sum(f.startswith("APPLICATION_TYPE_") for f in p.features) == 1


def test_split_scale_fits_scaler_on_train_only(df: pd.DataFrame) -> None:
    p = preprocess(df.head(4000))
    s = split_scale(p, seed=1, test_size=0.25)
    assert s.x_train.shape[0] == 3000 and s.x_test.shape[0] == 1000
    assert s.x_train.shape[1] == s.x_test.shape[1] == p.x.shape[1]
    # Train columns are standardised; test columns are transformed with the *train* statistics.
    means = s.x_train.mean(axis=0)
    assert np.allclose(means, 0, atol=1e-3)
    unscaled_test_mean = (s.x_test * s.scaler_scale + s.scaler_mean).mean(axis=0)
    assert not np.allclose(unscaled_test_mean, s.scaler_mean, atol=1e-6)
    # Stratified: class balance preserved within a point.
    assert abs(s.y_train.mean() - s.y_test.mean()) < 0.02
