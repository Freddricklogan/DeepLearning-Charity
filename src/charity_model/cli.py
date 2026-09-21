"""Command-line entry point: `charity-model report --out dist`."""

from __future__ import annotations

from pathlib import Path

import typer

from .network import NetConfig
from .report import write_report

app = typer.Typer(add_completion=False, help="Charity funding outcome classifier.")


@app.callback()
def main() -> None:
    """Charity funding outcome classifier."""


DEFAULT_DATA = Path("data") / "charity_data.csv.gz"


@app.command()
def report(
    out: Path = typer.Option(Path("dist"), help="Output directory for the static report."),
    data: Path = typer.Option(DEFAULT_DATA, help="Path to charity_data.csv or .csv.gz."),
    seed: int = typer.Option(42, help="Seed for the split and the network."),
    max_epochs: int = typer.Option(30, help="Upper bound on training epochs."),
) -> None:
    """Train the baseline and the network, evaluate, and write the report and model card."""
    cfg = NetConfig(max_epochs=max_epochs)
    path = write_report(out, data, seed=seed, cfg=cfg)
    print(f"wrote {path}")


if __name__ == "__main__":
    app()
