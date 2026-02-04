from __future__ import annotations

import pandas as pd

from fowsim.config.settings import Settings
from fowsim.simulation.scenarios import scenario_registry


def _make_future_frame(panel: pd.DataFrame, country: str, horizon: int) -> pd.DataFrame:
    """Create naive future feature frame by extending last observed year.
    Realistic future features can be modeled later; this is a working MVP.
    """
    dfc = panel[panel["iso3"] == country].sort_values("year").copy()
    if dfc.empty:
        raise ValueError(f"No data for country={country}")

    last = dfc.iloc[-1].to_dict()
    last_year = int(last["year"])
    future_years = list(range(last_year + 1, last_year + horizon + 1))

    rows = []
    for y in future_years:
        r = dict(last)
        r["year"] = y
        rows.append(r)

    future = pd.DataFrame(rows)
    # drop any lag/rolling columns if present (optional; keep for now)
    return future


def run_simulation(settings: Settings, country: str, scenario_name: str, horizon: int):
    panel = pd.read_parquet(settings.paths.processed_panel)
    scens = scenario_registry()
    if scenario_name not in scens:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {sorted(scens)}")

    future = _make_future_frame(panel, country=country, horizon=horizon)
    future_mod = scens[scenario_name].apply(future)

    # MVP output: save the scenario-adjusted future features (forecasting integration comes next)
    out = future_mod.copy()
    out["scenario"] = scenario_name

    out_path = settings.paths.data_processed / f"simulation_{country}_{scenario_name}_{horizon}.parquet"
    out.to_parquet(out_path, index=False)
    return out_path
