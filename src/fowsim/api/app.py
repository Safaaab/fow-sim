"""FastAPI REST API for Future of Work Simulator."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
import yaml

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(
        "FastAPI not installed. Install with: pip install -e '.[api]'"
    ) from e

from fowsim.config.settings import Settings
from fowsim.simulation.scenarios import scenario_registry
from fowsim.simulation.simulator import _make_future_frame


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FoW-Sim API",
    description="API for Future of Work Prediction and Scenario Simulation",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = Settings()
scenarios = scenario_registry()


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    version: str


class DataSummary(BaseModel):
    countries: int
    years_range: str
    indicators: int
    total_observations: int


class ForecastRequest(BaseModel):
    country: str = Field(..., description="ISO3 country code")
    target: str = Field(..., description="Target variable to forecast")
    horizon: int = Field(..., description="Forecast horizon in years")


class ScenarioRequest(BaseModel):
    country: str = Field(..., description="ISO3 country code")
    scenario: str = Field(..., description="Scenario name")
    horizon: int = Field(..., description="Simulation horizon in years")
    indicator: Optional[str] = Field(None, description="Specific indicator to return")


class ModelMetrics(BaseModel):
    model: str
    target: str
    horizon: int
    rmse: float
    mae: float


# Endpoints
@app.get("/", response_model=HealthResponse)
def root():
    """Root endpoint with API status."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/data/summary", response_model=DataSummary)
def get_data_summary():
    """Get summary statistics of the dataset."""
    try:
        panel = pd.read_parquet(settings.paths.processed_panel)
        
        countries = panel["iso3"].nunique()
        years = sorted(panel["year"].unique())
        years_range = f"{int(min(years))}-{int(max(years))}"
        indicators = len([c for c in panel.columns if c not in {"iso3", "year", "country"}])
        total_obs = len(panel)
        
        return {
            "countries": countries,
            "years_range": years_range,
            "indicators": indicators,
            "total_observations": total_obs
        }
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/countries")
def get_countries():
    """Get list of available countries."""
    try:
        panel = pd.read_parquet(settings.paths.processed_panel)
        countries = sorted(panel["iso3"].unique().tolist())
        return {"countries": countries, "count": len(countries)}
    except Exception as e:
        logger.error(f"Error getting countries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/indicators")
def get_indicators():
    """Get list of available indicators."""
    try:
        panel = pd.read_parquet(settings.paths.processed_panel)
        indicators = [c for c in panel.columns if c not in {"iso3", "year", "country"}]
        
        # Separate base indicators from engineered features
        base_indicators = [c for c in indicators if not any(x in c for x in ["_lag", "_roll"])]
        feature_indicators = [c for c in indicators if any(x in c for x in ["_lag", "_roll"])]
        
        return {
            "base_indicators": base_indicators,
            "engineered_features": feature_indicators,
            "total": len(indicators)
        }
    except Exception as e:
        logger.error(f"Error getting indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/country/{country}")
def get_country_data(country: str, indicator: Optional[str] = None):
    """Get historical data for a specific country."""
    try:
        panel = pd.read_parquet(settings.paths.processed_panel)
        
        if country not in panel["iso3"].values:
            raise HTTPException(status_code=404, detail=f"Country {country} not found")
        
        country_data = panel[panel["iso3"] == country].sort_values("year")
        
        if indicator:
            if indicator not in country_data.columns:
                raise HTTPException(status_code=404, detail=f"Indicator {indicator} not found")
            country_data = country_data[["year", indicator]]
        
        return country_data.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting country data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scenarios")
def get_scenarios():
    """Get list of available scenarios."""
    return {
        "scenarios": [
            {"name": name, "description": scenario.description}
            for name, scenario in scenarios.items()
        ],
        "count": len(scenarios)
    }


@app.post("/scenarios/simulate")
def simulate_scenario(request: ScenarioRequest):
    """Run scenario simulation for a country."""
    try:
        if request.scenario not in scenarios:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario '{request.scenario}' not found. Available: {list(scenarios.keys())}"
            )
        
        panel = pd.read_parquet(settings.paths.processed_panel)
        
        if request.country not in panel["iso3"].values:
            raise HTTPException(status_code=404, detail=f"Country {request.country} not found")
        
        # Generate future frame
        future = _make_future_frame(panel, country=request.country, horizon=request.horizon)
        
        # Apply scenario
        future_modified = scenarios[request.scenario].apply(future)
        
        # Filter by indicator if specified
        if request.indicator:
            if request.indicator not in future_modified.columns:
                raise HTTPException(status_code=404, detail=f"Indicator {request.indicator} not found")
            future_modified = future_modified[["year", request.indicator]]
        
        return {
            "country": request.country,
            "scenario": request.scenario,
            "horizon": request.horizon,
            "data": future_modified.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running scenario simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/metrics")
def get_model_metrics(
    target: Optional[str] = None,
    horizon: Optional[int] = None,
    model: Optional[str] = None
):
    """Get model performance metrics."""
    try:
        metrics = pd.read_csv(settings.paths.backtest_metrics)
        
        # Apply filters
        if target:
            metrics = metrics[metrics["target"] == target]
        if horizon:
            metrics = metrics[metrics["horizon"] == horizon]
        if model:
            metrics = metrics[metrics["model"] == model]
        
        if len(metrics) == 0:
            return {"metrics": [], "count": 0}
        
        return {
            "metrics": metrics.to_dict(orient="records"),
            "count": len(metrics)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model metrics not found. Run training first.")
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/best")
def get_best_models():
    """Get best performing model for each target-horizon combination."""
    try:
        metrics = pd.read_csv(settings.paths.backtest_metrics)
        
        # Group by target and horizon, find best model
        best_models = metrics.groupby(["target", "horizon"]).apply(
            lambda g: g.loc[g["rmse"].idxmin()]
        ).reset_index(drop=True)
        
        return {
            "best_models": best_models[["target", "horizon", "model", "rmse", "mae"]].to_dict(orient="records"),
            "count": len(best_models)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model metrics not found. Run training first.")
    except Exception as e:
        logger.error(f"Error getting best models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecasts")
def get_forecasts(
    country: Optional[str] = None,
    target: Optional[str] = None,
    horizon: Optional[int] = None
):
    """Get model forecasts."""
    try:
        forecasts = pd.read_parquet(settings.paths.processed_forecasts)
        
        # Apply filters
        if country:
            forecasts = forecasts[forecasts["iso3"] == country]
        if target:
            forecasts = forecasts[forecasts["target"] == target]
        if horizon:
            forecasts = forecasts[forecasts["horizon"] == horizon]
        
        if len(forecasts) == 0:
            return {"forecasts": [], "count": 0}
        
        return {
            "forecasts": forecasts.to_dict(orient="records"),
            "count": len(forecasts)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Forecasts not found. Run training first.")
    except Exception as e:
        logger.error(f"Error getting forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
