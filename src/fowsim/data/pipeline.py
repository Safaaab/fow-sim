from __future__ import annotations

import logging
import yaml
import pandas as pd
from pathlib import Path

from fowsim.config.settings import Settings
from fowsim.data.ingest_worldbank import fetch_indicator
from fowsim.data.build_features import add_lag_features, add_rolling_features
from fowsim.data.validate import validate_panel

logger = logging.getLogger(__name__)


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_external_dataset(data_raw: Path, filename: str, key_cols: list[str]) -> pd.DataFrame:
    """Load external CSV dataset from data/raw folder.
    
    Args:
        data_raw: Path to data/raw directory
        filename: Name of the CSV file
        key_cols: Columns to use for merging (e.g., ['iso3', 'year'])
    
    Returns:
        DataFrame with external data or empty DataFrame if not found
    """
    filepath = data_raw / filename
    if not filepath.exists():
        logger.warning(f"External dataset not found: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        # Ensure key columns exist
        if not all(col in df.columns for col in key_cols):
            logger.warning(f"Missing key columns in {filename}: {key_cols}")
            return pd.DataFrame()
        
        logger.info(f"Loaded external dataset: {filename} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return pd.DataFrame()


def _merge_external_datasets(panel: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Merge external datasets (automation risk, remote work, AI adoption, skills gap).
    
    As per project proposal, external datasets from Kaggle, IBM, and other sources
    provide additional features for prediction.
    """
    data_raw = settings.paths.data_raw
    
    # Load automation risk data (Frey & Osborne style)
    automation_df = _load_external_dataset(
        data_raw, "automation_risk.csv", ["iso3", "year"]
    )
    if not automation_df.empty:
        merge_cols = [c for c in automation_df.columns if c not in ["country"]]
        panel = panel.merge(
            automation_df[merge_cols], 
            on=["iso3", "year"], 
            how="left"
        )
        logger.info("Merged automation risk data")
    
    # Load remote work trends
    remote_df = _load_external_dataset(
        data_raw, "remote_work_trends.csv", ["iso3", "year"]
    )
    if not remote_df.empty:
        merge_cols = [c for c in remote_df.columns if c not in ["country"]]
        panel = panel.merge(
            remote_df[merge_cols], 
            on=["iso3", "year"], 
            how="left"
        )
        logger.info("Merged remote work trends data")
    
    # Load AI adoption index
    ai_df = _load_external_dataset(
        data_raw, "ai_adoption_index.csv", ["iso3", "year"]
    )
    if not ai_df.empty:
        merge_cols = [c for c in ai_df.columns if c not in ["country"]]
        panel = panel.merge(
            ai_df[merge_cols], 
            on=["iso3", "year"], 
            how="left"
        )
        logger.info("Merged AI adoption index data")
    
    # Load skills gap data
    skills_df = _load_external_dataset(
        data_raw, "skills_gap_data.csv", ["iso3", "year"]
    )
    if not skills_df.empty:
        merge_cols = [c for c in skills_df.columns if c not in ["country"]]
        panel = panel.merge(
            skills_df[merge_cols], 
            on=["iso3", "year"], 
            how="left"
        )
        logger.info("Merged skills gap data")
    
    return panel


def build_panel_dataset(settings: Settings, start_year: int, end_year: int) -> None:
    """Build complete panel dataset from World Bank and external sources.
    
    This function implements the data collection methodology from the project proposal:
    - World Bank API for core workforce and technology indicators
    - External datasets (Kaggle, IBM, OECD) for automation, remote work, AI adoption
    - Feature engineering with lags and rolling windows
    - Data validation and quality checks
    """
    cfg = _load_yaml(settings.root / "src" / "fowsim" / "config" / "indicators.yaml")
    targets = cfg.get("targets", {})
    features = cfg.get("features", {})
    countries = cfg.get("countries", [])
    if not countries:
        raise ValueError("No countries configured in indicators.yaml")

    logger.info(f"Building panel dataset for {len(countries)} countries, {start_year}-{end_year}")
    
    all_series = {**targets, **features}
    frames = []
    for name, code in all_series.items():
        wb = fetch_indicator(code, countries=countries, start_year=start_year, end_year=end_year)
        df = wb.df.rename(columns={"value": name})
        frames.append(df[["iso3", "year", name]])

    # Merge to panel
    panel = frames[0]
    for df in frames[1:]:
        panel = panel.merge(df, on=["iso3", "year"], how="outer")

    panel = panel.sort_values(["iso3", "year"])
    
    # Merge external datasets (automation risk, remote work, AI adoption, skills gap)
    logger.info("Merging external datasets...")
    panel = _merge_external_datasets(panel, settings)
    
    # Simple imputation: forward-fill within country, then backfill (kept conservative)
    panel = panel.groupby("iso3", sort=False).apply(lambda d: d.sort_values("year").ffill().bfill()).reset_index(drop=True)

    # Add features (lags and rolling windows for time-series forecasting)
    numeric_cols = [c for c in panel.columns if c not in {"iso3", "year", "country"}]
    panel = add_lag_features(panel, cols=numeric_cols, lags=[1, 2, 3])
    panel = add_rolling_features(panel, cols=numeric_cols, windows=[3, 5])

    validate_panel(panel)

    settings.paths.ensure()
    panel.to_parquet(settings.paths.processed_panel, index=False)
    
    logger.info(f"Panel dataset saved: {settings.paths.processed_panel}")
    logger.info(f"Shape: {panel.shape[0]} rows, {panel.shape[1]} columns")
