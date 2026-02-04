from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path

    processed_panel: Path
    processed_forecasts: Path
    backtest_metrics: Path

    def ensure(self) -> None:
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_interim.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # Root is the repo root (where pyproject.toml is located)
    # Path: settings.py -> config -> fowsim -> src -> fow-sim (root)
    root: Path = Path(__file__).resolve().parents[3]
    random_seed: int = 42

    def __post_init__(self) -> None:
        p = self.paths
        p.ensure()

    @property
    def paths(self) -> Paths:
        data_dir = self.root / "data"
        processed = data_dir / "processed"
        return Paths(
            root=self.root,
            data_raw=data_dir / "raw",
            data_interim=data_dir / "interim",
            data_processed=processed,
            processed_panel=processed / "panel.parquet",
            processed_forecasts=processed / "forecasts.parquet",
            backtest_metrics=processed / "backtest_metrics.csv",
        )
