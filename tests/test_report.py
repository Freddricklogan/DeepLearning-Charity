import json
from pathlib import Path

import pandas as pd

from charity_model.data import load
from charity_model.network import NetConfig
from charity_model.report import render_model_card, run, write_report

DATA = Path(__file__).resolve().parents[1] / "data" / "charity_data.csv.gz"


def test_write_report_produces_page_json_and_card(tmp_path: Path) -> None:
    small = tmp_path / "small.csv"
    load(DATA).head(2000).to_csv(small, index=False)
    cfg = NetConfig(hidden=(16,), max_epochs=2, patience=1, batch_size=128)
    out = write_report(tmp_path / "dist", small, seed=5, cfg=cfg)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html and "onclick" not in html and "style=" not in html
    assert (tmp_path / "dist" / "src" / "exec-shell.js").exists()
    data = json.loads((tmp_path / "dist" / "report.json").read_text())
    assert data["rows"] == 2000 and data["testRows"] == 500
    assert 0 <= data["network"]["accuracy"] <= 1 and 0 <= data["baseline"]["roc_auc"] <= 1
    card = (tmp_path / "dist" / "MODEL_CARD.md").read_text()
    assert "## Evaluation" in card and f"{data['network']['accuracy']:.4f}" in card
    assert "$" not in card.replace("$seed", "")  # every template field was substituted


def test_run_reports_binning_and_model_card_text(tmp_path: Path) -> None:
    small = tmp_path / "small.csv"
    pd.read_csv(DATA).head(1500).to_csv(small, index=False)
    r = run(small, seed=2, cfg=NetConfig(hidden=(8,), max_epochs=1, patience=1))
    assert r.trained.epochs_run == 1
    assert len(r.sweep) == 19 and len(r.calibration) >= 1
    card = render_model_card(r)
    assert "Logistic regression" in card and "--seed 2" in card
