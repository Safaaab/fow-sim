from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

import pandas as pd
import requests


logger = logging.getLogger(__name__)
WB_BASE = "https://api.worldbank.org/v2"
OECD_BASE = "https://stats.oecd.org/SDMX-JSON/data"


@dataclass(frozen=True)
class WBResult:
    indicator_code: str
    df: pd.DataFrame  # columns: ["country", "countryiso3code", "date", "value"]


def fetch_indicator(
    indicator_code: str,
    countries: Iterable[str],
    start_year: int,
    end_year: int,
    sleep_s: float = 0.5,
    timeout_s: int = 300,
    max_retries: int = 5,
) -> WBResult:
    """Fetch yearly data for an indicator from World Bank API.

    Returns a DataFrame with year as int and value as float.
    """
    country_str = ";".join(countries)
    url = f"{WB_BASE}/country/{country_str}/indicator/{indicator_code}"
    params = {
        "format": "json",
        "per_page": 20000,
        "date": f"{start_year}:{end_year}",
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {indicator_code} (attempt {attempt+1}/{max_retries})")
            r = requests.get(url, params=params, timeout=timeout_s)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                raise ValueError(f"No data returned for {indicator_code}")

            rows = payload[1]
            out = []
            for row in rows:
                out.append(
                    {
                        "country": row["country"]["value"],
                        "iso3": row.get("countryiso3code"),
                        "year": int(row["date"]),
                        "value": row["value"],
                    }
                )

            df = pd.DataFrame(out)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["iso3", "year"]).sort_values(["iso3", "year"])
            time.sleep(sleep_s)
            logger.info(f"Successfully fetched {indicator_code}: {len(df)} rows")
            return WBResult(indicator_code=indicator_code, df=df)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {indicator_code}: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (2 ** attempt)  # Longer exponential backoff: 5, 10, 20, 40 seconds
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise
    
    # Should never reach here, but satisfy type checker
    raise RuntimeError(f"Failed to fetch {indicator_code} after {max_retries} attempts")


def fetch_oecd_indicator(
    dataset: str,
    indicator: str,
    countries: Iterable[str],
    start_year: int,
    end_year: int,
    sleep_s: float = 0.2,
    timeout_s: int = 30,
) -> pd.DataFrame:
    """Fetch data from OECD API.
    
    Example datasets:
    - LFS: Labour Force Statistics
    - PAT_IPC: Patents by technology
    - RGRWTH: Regional Growth
    """
    try:
        country_str = "+".join(countries)
        url = f"{OECD_BASE}/{dataset}/{country_str}.{indicator}/all"
        params = {"startTime": start_year, "endTime": end_year, "format": "json"}
        
        r = requests.get(url, params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        
        # Parse OECD JSON structure (complex)
        observations = data.get("dataSets", [{}])[0].get("observations", {})
        structure = data.get("structure", {})
        
        rows = []
        for obs_key, obs_val in observations.items():
            indices = [int(i) for i in obs_key.split(":")]
            value = obs_val[0] if isinstance(obs_val, list) else obs_val
            rows.append({"key": obs_key, "value": value})
        
        df = pd.DataFrame(rows)
        df["indicator_code"] = indicator
        time.sleep(sleep_s)
        return df
    except Exception as e:
        logger.warning(f"OECD fetch failed for {dataset}/{indicator}: {e}")
        return pd.DataFrame()


def fetch_automation_risk_data(data_path: Path) -> pd.DataFrame:
    """Load automation risk dataset (Frey & Osborne style).
    
    Expected CSV columns: occupation, automation_probability, country, year
    This should be manually downloaded from Kaggle or academic sources.
    """
    if not data_path.exists():
        logger.warning(f"Automation risk data not found at {data_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    required = ["country", "year", "automation_risk"]
    if not all(c in df.columns for c in required):
        logger.error(f"Automation data missing required columns: {required}")
        return pd.DataFrame()
    
    return df[["country", "year", "automation_risk"]]


def fetch_remote_work_data(data_path: Path) -> pd.DataFrame:
    """Load remote work adoption data (e.g., from surveys, ILO reports).
    
    Expected columns: country, year, remote_work_percentage
    """
    if not data_path.exists():
        logger.warning(f"Remote work data not found at {data_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    if "remote_work_percentage" in df.columns:
        return df[["country", "year", "remote_work_percentage"]]
    return pd.DataFrame()


def fetch_ai_adoption_data(data_path: Path) -> pd.DataFrame:
    """Load AI adoption metrics (e.g., from McKinsey AI reports).
    
    Expected columns: country, year, ai_adoption_index
    """
    if not data_path.exists():
        logger.warning(f"AI adoption data not found at {data_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    if "ai_adoption_index" in df.columns:
        return df[["country", "year", "ai_adoption_index"]]
    return pd.DataFrame()
