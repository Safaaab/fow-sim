from __future__ import annotations

import pandas as pd

# Suppress future warnings for pandas operations
pd.set_option('future.no_silent_downcasting', True)


def validate_panel(panel: pd.DataFrame) -> None:
    required = {"iso3", "year"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")

    if panel["year"].isna().any():
        raise ValueError("Panel has NaN years")

    # Ensure monotonic per country
    def check_monotonic(s: pd.Series) -> bool:
        diff = s.diff()
        diff = diff.fillna(1).infer_objects(copy=False)
        return (diff <= 0).any()
    
    bad = (
        panel.sort_values(["iso3", "year"])
        .groupby("iso3")["year"]
        .apply(check_monotonic)
    )
    if bool(bad.any()):
        offenders = bad[bad].index.tolist()
        raise ValueError(f"Non-increasing years for: {offenders[:10]}")

    # Light missingness check (not strict)
    numeric = panel.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns found in panel")
