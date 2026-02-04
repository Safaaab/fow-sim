from __future__ import annotations

import pandas as pd


def add_lag_features(
    panel: pd.DataFrame,
    group_col: str = "iso3",
    time_col: str = "year",
    cols: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    if cols is None:
        # all numeric except identifiers
        cols = [c for c in panel.columns if c not in {group_col, time_col, "country"}]
    if lags is None:
        lags = [1, 2, 3]

    panel = panel.sort_values([group_col, time_col]).copy()
    g = panel.groupby(group_col, sort=False)
    for c in cols:
        for L in lags:
            panel[f"{c}_lag{L}"] = g[c].shift(L)
    return panel


def add_rolling_features(
    panel: pd.DataFrame,
    group_col: str = "iso3",
    time_col: str = "year",
    cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if cols is None:
        cols = [c for c in panel.columns if c not in {group_col, time_col, "country"}]
    if windows is None:
        windows = [3, 5]

    panel = panel.sort_values([group_col, time_col]).copy()
    g = panel.groupby(group_col, sort=False)
    for c in cols:
        for w in windows:
            panel[f"{c}_rollmean{w}"] = g[c].rolling(w, min_periods=max(2, w // 2)).mean().reset_index(level=0, drop=True)
    return panel
