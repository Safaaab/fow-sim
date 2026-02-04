from __future__ import annotations

import pandas as pd


def rolling_origin_splits(
    panel: pd.DataFrame,
    time_col: str = "year",
    min_train_years: int = 3,  # Reduced to 3 to work with long-horizon forecasts (h=20)
    step: int = 1,
    max_splits: int = 8,
) -> list[tuple[int, int]]:
    """Return list of (train_end_year, test_year) for rolling-origin backtest.

    Example: train up to 2012 -> test 2013, then train up to 2013 -> test 2014, ...
    """
    years = sorted(panel[time_col].dropna().unique().tolist())
    if len(years) < (min_train_years + 1):
        return []

    splits = []
    start_idx = min_train_years - 1
    for i in range(start_idx, len(years) - 1, step):
        train_end = years[i]
        test_year = years[i + 1]
        splits.append((int(train_end), int(test_year)))
        if len(splits) >= max_splits:
            break
    return splits


def simple_train_test_split(
    panel: pd.DataFrame,
    time_col: str = "year",
    train_ratio: float = 0.6,
) -> tuple[list[int], list[int]]:
    """Simple train/test split for limited year datasets (e.g., h=20 horizon).
    
    Returns (train_years, test_years) where train_years is first 60% of years
    and test_years is remaining 40%.
    """
    years = sorted(panel[time_col].dropna().unique().tolist())
    if len(years) < 2:
        return [], []
    
    split_idx = max(1, int(len(years) * train_ratio))
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    # Ensure at least one test year
    if not test_years and len(years) > 1:
        train_years = years[:-1]
        test_years = [years[-1]]
    
    return train_years, test_years
