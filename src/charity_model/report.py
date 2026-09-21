"""Static HTML report and model card — the Pages artefact. Every number is computed in this run
from the vendored dataset and embedded as JSON for the Executive Shell. No inline script/style."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .baseline import fit_predict
from .data import Prepared, Split, load, preprocess, split_scale
from .evaluate import Metrics, metrics, reliability, roc_points, threshold_sweep
from .network import NetConfig, Trained, train_predict

PKG = Path(__file__).parent
SHELL_DIR = PKG / "shell"
TEMPLATES = PKG / "templates"
DATA_SOURCE = (
    "Alphabet Soup charity applications (34,299 rows) as distributed with the Deep Learning "
    "bootcamp exercise; vendored in data/charity_data.csv.gz"
)


@dataclass(frozen=True)
class RunResult:
    prepared: Prepared
    split: Split
    cfg: NetConfig
    seed: int
    baseline: Metrics
    network: Metrics
    trained: Trained
    baseline_roc: list[tuple[float, float]]
    network_roc: list[tuple[float, float]]
    calibration: list[tuple[float, float, int]]
    sweep: list[Metrics]


def run(data: Path, seed: int = 42, cfg: NetConfig | None = None) -> RunResult:
    cfg = cfg or NetConfig()
    p = preprocess(load(data))
    s = split_scale(p, seed=seed)
    base_prob = fit_predict(s.x_train, s.y_train, s.x_test, seed=seed)
    t = train_predict(s.x_train, s.y_train, s.x_test, s.y_test, cfg, seed=seed)
    return RunResult(
        prepared=p,
        split=s,
        cfg=cfg,
        seed=seed,
        baseline=metrics(s.y_test, base_prob),
        network=metrics(s.y_test, t.prob),
        trained=t,
        baseline_roc=roc_points(s.y_test, base_prob),
        network_roc=roc_points(s.y_test, t.prob),
        calibration=reliability(s.y_test, t.prob),
        sweep=threshold_sweep(s.y_test, t.prob),
    )


def _pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%"


def _row(cells: list[str], head: bool = False) -> str:
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _metrics_table(base: Metrics, net: Metrics) -> str:
    rows = [_row(["Metric", "Logistic regression", "Network"], head=True)]
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC AUC", "roc_auc"),
    ]:
        b = getattr(base, key)
        n = getattr(net, key)
        rows.append(_row([label, f"{b:.4f}", f"{n:.4f}"]))
    rows.append(
        _row(
            [
                "Confusion (TP / FP / TN / FN)",
                f"{base.tp} / {base.fp} / {base.tn} / {base.fn}",
                f"{net.tp} / {net.fp} / {net.tn} / {net.fn}",
            ]
        )
    )
    rows.append(
        _row(
            [
                "Majority-class accuracy",
                _pct(max(base.base_rate, 1 - base.base_rate)),
                "same test set",
            ]
        )
    )
    return f"<table>{''.join(rows)}</table>"


def _lines(
    series: list[tuple[str, list[tuple[float, float]], str]],
    x_label: str,
    y_label: str,
    width: int = 640,
    height: int = 240,
    y_range: tuple[float, float] | None = None,
) -> str:
    pad = 40
    xs = [x for _, pts, _ in series for x, _ in pts]
    ys = [y for _, pts, _ in series for _, y in pts]
    x_lo, x_hi = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y_lo, y_hi = y_range if y_range else (min(ys), max(ys) if max(ys) > min(ys) else min(ys) + 1)

    def sx(x: float) -> float:
        return pad + (x - x_lo) / (x_hi - x_lo) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - y_lo) / (y_hi - y_lo) * (height - 2 * pad)

    paths = []
    for label, pts, cls in series:
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        paths.append(
            f'<polyline class="{cls}" points="{d}"><title>{html.escape(label)}</title></polyline>'
        )
    bottom = height - pad
    axis = f'<path class="cm-axis" d="M{pad},{pad} L{pad},{bottom} L{width - pad},{bottom}"/>'
    tick = '<text class="cm-tick" x="{x}" y="{y}" text-anchor="{a}">{t}</text>'
    labels = (
        tick.format(x=pad - 4, y=pad + 4, a="end", t=f"{y_hi:.2f}")
        + tick.format(x=pad - 4, y=bottom + 4, a="end", t=f"{y_lo:.2f}")
        + tick.format(x=width - pad, y=bottom + 14, a="end", t=html.escape(x_label))
        + tick.format(x=pad, y=pad - 8, a="start", t=html.escape(y_label))
    )
    legend = " ".join(
        f'<tspan class="{cls}-t">{html.escape(label)}</tspan>' for label, _, cls in series
    )
    aria = html.escape(f"{y_label} against {x_label}")
    return (
        f'<svg class="cm-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{aria}">'
        f"{axis}{''.join(paths)}{labels}"
        + tick.format(x=width - pad, y=pad - 8, a="end", t=legend)
        + "</svg>"
    )


def _calibration_table(rows: list[tuple[float, float, int]]) -> str:
    out = [_row(["Mean predicted", "Observed rate", "Rows"], head=True)]
    out += [_row([f"{p:.3f}", f"{o:.3f}", str(n)]) for p, o, n in rows]
    return f"<table>{''.join(out)}</table>"


def _sweep_table(sweep: list[Metrics]) -> str:
    out = [_row(["Threshold", "Accuracy", "Precision", "Recall", "F1"], head=True)]
    out += [
        _row(
            [
                f"{m.threshold:.2f}",
                f"{m.accuracy:.4f}",
                f"{m.precision:.4f}",
                f"{m.recall:.4f}",
                f"{m.f1:.4f}",
            ]
        )
        for m in sweep
        if round(m.threshold * 100) % 10 == 0 or abs(m.threshold - 0.5) < 1e-9
    ]
    return f"<table>{''.join(out)}</table>"


def _binned_list(binned: dict[str, list[str]]) -> str:
    return "; ".join(
        f"<strong>{html.escape(col)}</strong>: {len(vals)} values folded into Other"
        for col, vals in binned.items()
    )


def kpi_json(r: RunResult) -> dict[str, object]:
    return {
        "rows": int(r.prepared.x.shape[0]),
        "features": len(r.prepared.features),
        "testRows": int(r.split.y_test.size),
        "baseRate": r.baseline.base_rate,
        "baseline": r.baseline.as_dict(),
        "network": r.network.as_dict(),
        "epochsRun": r.trained.epochs_run,
        "bestEpoch": r.trained.best_epoch,
        "params": r.trained.params,
        "hidden": list(r.cfg.hidden),
        "seed": r.seed,
    }


def render_html(r: RunResult, pages: str) -> str:
    hist = r.trained.history
    epochs = list(range(1, r.trained.epochs_run + 1))
    curve = _lines(
        [
            (
                "Training loss",
                list(zip(map(float, epochs), hist["loss"], strict=True)),
                "cm-line-a",
            ),
            (
                "Validation loss",
                list(zip(map(float, epochs), hist["val_loss"], strict=True)),
                "cm-line-b",
            ),
        ],
        "Epoch",
        "Binary cross-entropy",
    )
    roc = _lines(
        [
            ("Network", r.network_roc, "cm-line-a"),
            ("Logistic regression", r.baseline_roc, "cm-line-b"),
            ("Chance", [(0.0, 0.0), (1.0, 1.0)], "cm-line-c"),
        ],
        "False-positive rate",
        "True-positive rate",
        y_range=(0.0, 1.0),
    )
    tpl = Template((TEMPLATES / "page.html").read_text(encoding="utf-8"))
    return tpl.substitute(
        pages=html.escape(pages),
        source=html.escape(DATA_SOURCE),
        rows=f"{r.prepared.x.shape[0]:,}",
        features=str(len(r.prepared.features)),
        test_rows=f"{r.split.y_test.size:,}",
        base_rate=_pct(r.baseline.base_rate),
        binned=_binned_list(r.prepared.binned),
        hidden=" → ".join(str(h) for h in r.cfg.hidden),
        dropout=str(r.cfg.dropout),
        params=f"{r.trained.params:,}",
        epochs=str(r.trained.epochs_run),
        best_epoch=str(r.trained.best_epoch),
        max_epochs=str(r.cfg.max_epochs),
        patience=str(r.cfg.patience),
        net_acc=_pct(r.network.accuracy, 2),
        base_acc=_pct(r.baseline.accuracy, 2),
        net_auc=f"{r.network.roc_auc:.3f}",
        base_auc=f"{r.baseline.roc_auc:.3f}",
        gap=f"{(r.network.accuracy - r.baseline.accuracy) * 100:+.2f}",
        metrics_table=_metrics_table(r.baseline, r.network),
        curve=curve,
        roc=roc,
        calibration=_calibration_table(r.calibration),
        sweep=_sweep_table(r.sweep),
        seed=str(r.seed),
        report_json=json.dumps(kpi_json(r)),
    )


def render_model_card(r: RunResult) -> str:
    n, b = r.network, r.baseline
    tpl = Template((TEMPLATES / "model_card.md").read_text(encoding="utf-8"))
    return tpl.substitute(
        seed=str(r.seed),
        hidden=" → ".join(str(h) for h in r.cfg.hidden),
        dropout=str(r.cfg.dropout),
        params=f"{r.trained.params:,}",
        lr=str(r.cfg.learning_rate),
        batch=str(r.cfg.batch_size),
        max_epochs=str(r.cfg.max_epochs),
        patience=str(r.cfg.patience),
        epochs=str(r.trained.epochs_run),
        best_epoch=str(r.trained.best_epoch),
        source=DATA_SOURCE,
        rows=f"{r.prepared.x.shape[0]:,}",
        binned="; ".join(f"{c}: {len(v)} values" for c, v in r.prepared.binned.items()),
        features=str(len(r.prepared.features)),
        test_rows=f"{r.split.y_test.size:,}",
        base_rate=f"{b.base_rate:.4f}",
        b_acc=f"{b.accuracy:.4f}",
        n_acc=f"{n.accuracy:.4f}",
        b_prec=f"{b.precision:.4f}",
        n_prec=f"{n.precision:.4f}",
        b_rec=f"{b.recall:.4f}",
        n_rec=f"{n.recall:.4f}",
        b_f1=f"{b.f1:.4f}",
        n_f1=f"{n.f1:.4f}",
        b_auc=f"{b.roc_auc:.4f}",
        n_auc=f"{n.roc_auc:.4f}",
        majority=f"{max(b.base_rate, 1 - b.base_rate):.4f}",
        gap=f"{(n.accuracy - b.accuracy) * 100:+.2f}",
    )


def write_report(
    out: Path,
    data: Path,
    seed: int = 42,
    cfg: NetConfig | None = None,
    pages: str = "https://freddricklogan.github.io/DeepLearning-Charity/",
) -> Path:
    r = run(data, seed=seed, cfg=cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    shutil.copy(SHELL_DIR / "exec-shell.css", out / "src" / "exec-shell.css")
    shutil.copy(SHELL_DIR / "exec-shell.js", out / "src" / "exec-shell.js")
    shutil.copy(PKG / "report.js", out / "src" / "report.js")
    shutil.copy(TEMPLATES / "report.css", out / "src" / "report.css")
    (out / "index.html").write_text(render_html(r, pages), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(kpi_json(r), indent=2), encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(render_model_card(r), encoding="utf-8")
    return out / "index.html"
