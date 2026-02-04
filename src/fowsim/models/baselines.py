from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd

# Suppress statsmodels warnings about index
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*unsupported index.*')
warnings.filterwarnings('ignore', message='.*No supported index.*')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


def ets_forecast(series: pd.Series, steps: int) -> pd.Series:
    """Simple ETS (Exponential Smoothing) forecast for one univariate series.
    
    Args:
        series: Time series data
        steps: Number of steps to forecast
    
    Returns:
        Forecasted series
    """
    s = series.dropna().astype(float)
    if len(s) < 4:
        # fallback: last value for very limited data
        last = float(s.iloc[-1]) if len(s) else float("nan")
        return pd.Series([last] * steps)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Use simpler model for limited data
            if len(s) < 8:
                model = ExponentialSmoothing(
                    s,
                    trend=None,
                    seasonal=None,
                    initialization_method="estimated"
                )
            else:
                model = ExponentialSmoothing(
                    s,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated"
                )
            fit = model.fit(optimized=True)
            fc = fit.forecast(steps)
            return pd.Series(fc.values)
    except Exception:
        # Silently fall back to naive forecast
        last = float(s.iloc[-1])
        return pd.Series([last] * steps)


def arima_forecast(
    series: pd.Series,
    steps: int,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (0, 0, 0, 0)
) -> pd.Series:
    """ARIMA/SARIMA forecast for univariate time series.
    
    Args:
        series: Time series data
        steps: Number of steps to forecast
        order: (p, d, q) order for ARIMA
        seasonal_order: (P, D, Q, s) seasonal order
    
    Returns:
        Forecasted series
    """
    s = series.dropna().astype(float)
    # Minimum 6 data points for ARIMA (reduced from 12 to handle limited data)
    if len(s) < 6:
        # Silently fall back to ETS - no warning needed
        return ets_forecast(series, steps)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if seasonal_order[3] > 0:
                # SARIMA
                model = SARIMAX(
                    s,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                # ARIMA - use simpler order for limited data
                if len(s) < 10:
                    order = (1, 0, 0)  # Simple AR(1) for limited data
                model = ARIMA(s, order=order, enforce_stationarity=False)
            
            fit = model.fit(disp=False)
            fc = fit.forecast(steps=steps)
            return pd.Series(fc.values if hasattr(fc, 'values') else fc)
    
    except Exception as e:
        # Silently fall back to ETS - no warning needed
        return ets_forecast(series, steps)


def naive_forecast(series: pd.Series, steps: int) -> pd.Series:
    """Naive forecast: last observed value repeated.
    
    Simple baseline that's hard to beat for random walk series.
    """
    s = series.dropna().astype(float)
    if len(s) == 0:
        return pd.Series([np.nan] * steps)
    
    last = float(s.iloc[-1])
    return pd.Series([last] * steps)


def drift_forecast(series: pd.Series, steps: int) -> pd.Series:
    """Drift forecast: linear trend from first to last observation.
    
    Good baseline for series with clear trends.
    """
    s = series.dropna().astype(float)
    if len(s) < 2:
        return naive_forecast(series, steps)
    
    first_val = float(s.iloc[0])
    last_val = float(s.iloc[-1])
    n = len(s)
    
    # Calculate drift
    drift = (last_val - first_val) / (n - 1)
    
    # Forecast
    forecasts = [last_val + drift * (i + 1) for i in range(steps)]
    return pd.Series(forecasts)


def seasonal_naive_forecast(series: pd.Series, steps: int, season_length: int = 12) -> pd.Series:
    """Seasonal naive forecast: repeat seasonal pattern.
    
    Args:
        series: Time series data
        steps: Number of steps to forecast
        season_length: Length of seasonal period (e.g., 12 for monthly data)
    
    Returns:
        Forecasted series
    """
    s = series.dropna().astype(float)
    if len(s) < season_length:
        return naive_forecast(series, steps)
    
    # Use last season as forecast
    last_season = s.iloc[-season_length:].values
    forecasts = []
    
    for i in range(steps):
        forecasts.append(last_season[i % season_length])
    
    return pd.Series(forecasts)


def moving_average_forecast(series: pd.Series, steps: int, window: int = 5) -> pd.Series:
    """Moving average forecast.
    
    Args:
        series: Time series data
        steps: Number of steps to forecast
        window: Window size for moving average
    
    Returns:
        Forecasted series
    """
    s = series.dropna().astype(float)
    if len(s) < window:
        return naive_forecast(series, steps)
    
    # Calculate moving average of last window
    ma = s.iloc[-window:].mean()
    return pd.Series([ma] * steps)


def ensemble_baseline_forecast(series: pd.Series, steps: int) -> pd.Series:
    """Ensemble of baseline methods for robust forecasting.
    
    Combines naive, drift, and ETS forecasts with equal weights.
    """
    forecasts = []
    
    # Get individual forecasts
    try:
        naive_fc = naive_forecast(series, steps)
        forecasts.append(naive_fc)
    except:
        pass
    
    try:
        drift_fc = drift_forecast(series, steps)
        forecasts.append(drift_fc)
    except:
        pass
    
    try:
        ets_fc = ets_forecast(series, steps)
        forecasts.append(ets_fc)
    except:
        pass
    
    if not forecasts:
        return pd.Series([np.nan] * steps)
    
    # Average forecasts
    ensemble = pd.concat(forecasts, axis=1).mean(axis=1)
    return ensemble
