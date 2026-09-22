"""Interactive companion to the static report.

Run: `uv run --extra app streamlit run streamlit_app.py`.

Trains the baseline and the network once (cached), then lets you move the decision threshold and
watch precision, recall and the confusion counts change on the held-out rows.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

# Community Cloud runs this file from a plain checkout; make the src/ package importable there.
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from charity_model.baseline import fit_predict
from charity_model.data import load, preprocess, split_scale
from charity_model.evaluate import metrics
from charity_model.network import NetConfig, train_predict

DATA = Path("data") / "charity_data.csv.gz"

st.set_page_config(page_title="Charity funding classifier", layout="wide")
st.title("Charity funding outcome classifier")
st.caption("Alphabet Soup dataset, vendored. Every number is computed in this session.")


@st.cache_resource
def fit(seed: int):  # type: ignore[no-untyped-def]
    split = split_scale(preprocess(load(DATA)), seed=seed)
    base = fit_predict(split.x_train, split.y_train, split.x_test, seed=seed)
    net = train_predict(split.x_train, split.y_train, split.x_test, split.y_test, NetConfig(), seed)
    return split, base, net.prob, net


seed = st.sidebar.number_input("Seed", value=42, step=1)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
split, base_prob, net_prob, trained = fit(int(seed))

b = metrics(split.y_test, base_prob, threshold)
n = metrics(split.y_test, net_prob, threshold)
c1, c2 = st.columns(2)
panels = ((c1, "Logistic regression", b), (c2, f"Network ({trained.epochs_run} epochs)", n))
for col, name, m in panels:
    col.subheader(name)
    col.metric("Accuracy", f"{m.accuracy:.4f}")
    col.metric("Precision", f"{m.precision:.4f}")
    col.metric("Recall", f"{m.recall:.4f}")
    col.metric("F1", f"{m.f1:.4f}")
    col.metric("ROC AUC", f"{m.roc_auc:.4f}")
    col.write(f"TP {m.tp} · FP {m.fp} · TN {m.tn} · FN {m.fn}")
st.write(
    f"{split.y_test.size:,} held-out rows; positive rate {b.base_rate:.4f}. "
    "The AUC does not move with the threshold; everything else does."
)
