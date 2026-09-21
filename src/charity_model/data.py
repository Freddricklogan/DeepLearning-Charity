"""Loading and preprocessing for the Alphabet Soup charity dataset.

Every transformation is a pure function so the test suite can pin its behaviour: rare-category
binning with explicit thresholds, identifier columns dropped, a log transform on the heavy-tailed
ask amount, one-hot encoding, and a scaler fitted on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "IS_SUCCESSFUL"
IDENTIFIERS = ("EIN", "NAME")
CATEGORICAL = (
    "APPLICATION_TYPE",
    "AFFILIATION",
    "CLASSIFICATION",
    "USE_CASE",
    "ORGANIZATION",
    "INCOME_AMT",
    "SPECIAL_CONSIDERATIONS",
)
NUMERIC = ("STATUS", "ASK_AMT")
REQUIRED = (*IDENTIFIERS, *CATEGORICAL, *NUMERIC, TARGET)
DEFAULT_THRESHOLDS: dict[str, int] = {"APPLICATION_TYPE": 500, "CLASSIFICATION": 1000}

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Prepared:
    """Encoded feature matrix with the names of its columns and the binning that produced it."""

    x: FloatArray
    y: IntArray
    features: list[str]
    binned: dict[str, list[str]]  # column -> categories folded into "Other"


@dataclass(frozen=True)
class Split:
    x_train: FloatArray
    x_test: FloatArray
    y_train: IntArray
    y_test: IntArray
    scaler_mean: FloatArray
    scaler_scale: FloatArray


def load(path: Path) -> pd.DataFrame:
    """Read the CSV (plain or gzip) and check the schema; raise on missing columns."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        msg = f"dataset is missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    if not set(df[TARGET].unique()) <= {0, 1}:
        msg = f"{TARGET} must be 0/1"
        raise ValueError(msg)
    return df


def bin_rare(
    series: pd.Series, threshold: int, other: str = "Other"
) -> tuple[pd.Series, list[str]]:
    """Fold categories that appear fewer than `threshold` times into `other`."""
    counts = series.value_counts()
    rare = sorted(str(c) for c in counts[counts < threshold].index)
    if not rare:
        return series.astype(str), []
    return series.astype(str).where(~series.astype(str).isin(rare), other), rare


def preprocess(df: pd.DataFrame, thresholds: dict[str, int] | None = None) -> Prepared:
    """Drop identifiers, bin rare categories, log-transform the ask amount, one-hot encode."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    work = df.drop(columns=list(IDENTIFIERS)).copy()
    binned: dict[str, list[str]] = {}
    for col, t in th.items():
        work[col], rare = bin_rare(work[col], t)
        if rare:
            binned[col] = rare
    for col in CATEGORICAL:
        work[col] = work[col].astype(str)
    # ASK_AMT spans 5,000 to 8.6e9; log1p keeps the scaler from being dominated by a few rows.
    work["ASK_AMT"] = np.log1p(work["ASK_AMT"].astype(float))
    y = work.pop(TARGET).to_numpy(dtype=np.int64)
    encoded = pd.get_dummies(work, columns=list(CATEGORICAL), dtype=np.float32)
    x = encoded.to_numpy(dtype=np.float32)
    return Prepared(x=x, y=y, features=[str(c) for c in encoded.columns], binned=binned)


def split_scale(p: Prepared, seed: int = 42, test_size: float = 0.25) -> Split:
    """Stratified split, then a StandardScaler fitted on the training rows only (no leakage)."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        p.x, p.y, test_size=test_size, random_state=seed, stratify=p.y
    )
    scaler = StandardScaler().fit(x_tr)
    return Split(
        x_train=scaler.transform(x_tr).astype(np.float32),
        x_test=scaler.transform(x_te).astype(np.float32),
        y_train=np.asarray(y_tr, dtype=np.int64),
        y_test=np.asarray(y_te, dtype=np.int64),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
    )
