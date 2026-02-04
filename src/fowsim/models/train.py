from __future__ import annotations

import logging
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Suppress statsmodels and sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*unsupported index.*')
warnings.filterwarnings('ignore', message='.*No supported index.*')
warnings.filterwarnings('ignore', category=FutureWarning)

from fowsim.config.settings import Settings
from fowsim.models.backtest import rolling_origin_splits, simple_train_test_split
from fowsim.models.metrics import rmse, mae, r2_score, accuracy_percentage
from fowsim.models.ml_models import make_models, create_ensemble_model
from fowsim.models.baselines import arima_forecast, ets_forecast

logger = logging.getLogger(__name__)


def _make_supervised(panel: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Create supervised dataset: predict target at t+h from features at t."""
    df = panel.sort_values(["iso3", "year"]).copy()
    df[f"{target}_t_plus_{horizon}"] = df.groupby("iso3", sort=False)[target].shift(-horizon)
    df = df.dropna(subset=[f"{target}_t_plus_{horizon}"])
    return df


def calculate_confidence_intervals(
    predictions: np.ndarray,
    actuals: np.ndarray,
    confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate confidence intervals based on prediction errors.
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    errors = actuals - predictions
    std_error = np.std(errors)
    
    # Z-score for confidence level
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    margin = z_score * std_error
    lower = predictions - margin
    upper = predictions + margin
    
    return lower, upper


def train_and_backtest(
    settings: Settings,
    horizons: list[int],
    include_baseline: bool = True,
    include_ensemble: bool = True
) -> None:
    """Train models and perform backtesting with comprehensive evaluation.
    
    Args:
        settings: Project settings
        horizons: Forecast horizons (e.g., [5, 10, 20] years)
        include_baseline: Include ARIMA/ETS baselines
        include_ensemble: Create ensemble models
    """
    panel = pd.read_parquet(settings.paths.processed_panel)
    cfg_path = settings.root / "src" / "fowsim" / "config" / "indicators.yaml"
    
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    targets: dict = cfg.get("targets", {})

    if not targets:
        raise ValueError("No targets found in indicators.yaml")

    # Features: all numeric columns except year/identifiers and future targets
    id_cols = {"iso3", "year", "country"}
    metrics_rows = []
    forecast_rows = []
    
    logger.info(f"Training models for {len(targets)} targets and {len(horizons)} horizons")

    for target_name in targets.keys():
        logger.info(f"Processing target: {target_name}")
        
        for h in horizons:
            logger.info(f"  Horizon: {h} years")
            ds = _make_supervised(panel, target=target_name, horizon=h)

            ycol = f"{target_name}_t_plus_{h}"
            y = ds[ycol]
            X = ds.drop(columns=[ycol] + [c for c in ds.columns if c in id_cols], errors="ignore")

            # Keep only numeric
            X = X.select_dtypes(include="number").fillna(0.0)

            # Get available years for this horizon
            available_years = sorted(ds["year"].unique())
            
            # Use rolling_origin_splits for normal cases, simple split for limited years (h=20)
            splits = rolling_origin_splits(ds, time_col="year", min_train_years=3)
            use_simple_split = (len(splits) == 0 or (h == 20 and len(available_years) <= 5))
            
            if use_simple_split:
                # For h=20 with limited years, use simple train/test split
                train_years, test_years = simple_train_test_split(ds, time_col="year", train_ratio=0.6)
                if not train_years or not test_years:
                    logger.warning(f"  No valid split for {target_name} h={h}")
                    continue
                splits = [(max(train_years), test_years[0])]  # Single split
                logger.info(f"  Using simple split: train {train_years}, test {test_years}")
            
            if not splits:
                logger.warning(f"  No valid splits for {target_name} h={h}")
                continue

            # Get ML models
            ml_models = make_models(settings.random_seed)
            
            # Add ensemble if requested
            if include_ensemble and len(ml_models) >= 3:
                ensemble = create_ensemble_model(ml_models, settings.random_seed)
                ml_models.append(ensemble)

            # Train and evaluate each model
            for bundle in ml_models:
                split_scores = []
                all_predictions = []
                all_actuals = []
                
                for train_end, test_year in splits:
                    # For simple split (h=20), use all train years
                    if use_simple_split:
                        train_mask = ds["year"].isin(train_years)
                        test_mask = ds["year"].isin(test_years)
                    else:
                        train_mask = ds["year"] <= train_end
                        test_mask = ds["year"] == test_year

                    Xtr, ytr = X[train_mask], y[train_mask]
                    Xte, yte = X[test_mask], y[test_mask]
                    if len(Xte) == 0 or len(Xtr) < 10:
                        continue

                    try:
                        model = bundle.model
                        model.fit(Xtr, ytr)
                        pred = model.predict(Xte)
                        
                        all_predictions.extend(pred)
                        all_actuals.extend(yte.values)

                        split_scores.append({
                            "target": target_name,
                            "horizon": h,
                            "model": bundle.name,
                            "train_end_year": train_end,
                            "test_year": test_year,
                            "rmse": rmse(yte, pred),
                            "mae": mae(yte, pred),
                            "r2": r2_score(yte, pred),
                            "accuracy_pct": accuracy_percentage(yte, pred, tolerance=0.1),
                            "n_test": len(yte),
                        })
                    except Exception as e:
                        logger.warning(f"  Model {bundle.name} failed on split {train_end}-{test_year}: {e}")
                        continue

                if split_scores:
                    # Calculate overall metrics with confidence intervals
                    try:
                        all_pred_arr = np.array(all_predictions)
                        all_act_arr = np.array(all_actuals)
                        lower, upper = calculate_confidence_intervals(all_pred_arr, all_act_arr)
                        ci_width = np.mean(upper - lower)
                    except:
                        ci_width = np.nan
                    
                    # Add summary metrics
                    for score in split_scores:
                        score["ci_width"] = ci_width
                    
                    metrics_rows.extend(split_scores)

            # Baseline models (ARIMA/ETS)
            if include_baseline:
                for baseline_name, baseline_func in [("arima", arima_forecast), ("ets", ets_forecast)]:
                    split_scores = []
                    
                    for train_end, test_year in splits:
                        train_mask = ds["year"] <= train_end
                        test_mask = ds["year"] == test_year
                        
                        # Train on time series per country
                        for country in ds["iso3"].unique():
                            country_train = ds[(ds["iso3"] == country) & train_mask]
                            country_test = ds[(ds["iso3"] == country) & test_mask]
                            
                            if len(country_train) < 10 or len(country_test) == 0:
                                continue
                            
                            try:
                                series = country_train.set_index("year")[target_name]
                                pred = baseline_func(series, steps=h)
                                actual = country_test[target_name].values
                                
                                if len(pred) > 0 and len(actual) > 0:
                                    split_scores.append({
                                        "target": target_name,
                                        "horizon": h,
                                        "model": f"{baseline_name}_baseline",
                                        "train_end_year": train_end,
                                        "test_year": test_year,
                                        "rmse": rmse(actual[:len(pred)], pred[:len(actual)]),
                                        "mae": mae(actual[:len(pred)], pred[:len(actual)]),
                                        "n_test": len(actual),
                                        "ci_width": np.nan,
                                    })
                            except Exception as e:
                                logger.debug(f"  Baseline {baseline_name} failed for {country}: {e}")
                                continue
                    
                    if split_scores:
                        metrics_rows.extend(split_scores)

            # Train final model on all data for this target+horizon
            mdf = pd.DataFrame([r for r in metrics_rows if r["target"] == target_name and r["horizon"] == h])
            if mdf.empty:
                logger.warning(f"  No metrics for {target_name} h={h}")
                continue
            
            # Select best model by average RMSE
            best_model_name = mdf.groupby("model")["rmse"].mean().sort_values().index[0]
            logger.info(f"  Best model for {target_name} h={h}: {best_model_name}")
            
            # Train final model on full data
            try:
                best_bundles = [b for b in ml_models if b.name == best_model_name]
                if not best_bundles:
                    continue
                    
                best = best_bundles[0].model
                best.fit(X, y)

                # Generate predictions
                pred_all = best.predict(X)
                
                # Calculate confidence intervals
                lower_ci, upper_ci = calculate_confidence_intervals(pred_all, y.values)

                # Save in-sample fitted forecast
                tmp = ds[["iso3", "year"]].copy()
                tmp["target"] = target_name
                tmp["horizon"] = h
                tmp["y_true"] = y.values
                tmp["y_pred"] = pred_all
                tmp["lower_ci"] = lower_ci
                tmp["upper_ci"] = upper_ci
                tmp["model"] = best_model_name
                forecast_rows.extend(tmp.to_dict("records"))
            except Exception as e:
                logger.error(f"  Final model training failed for {target_name} h={h}: {e}")

    # Save results
    out_metrics = pd.DataFrame(metrics_rows)
    out_forecasts = pd.DataFrame(forecast_rows)

    settings.paths.ensure()
    
    if not out_metrics.empty:
        out_metrics.to_csv(settings.paths.backtest_metrics, index=False)
        logger.info(f"Saved metrics to {settings.paths.backtest_metrics}")
        
        # Print summary
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        summary = out_metrics.groupby(["target", "horizon", "model"]).agg({
            "rmse": "mean",
            "mae": "mean",
        }).round(3)
        print(summary.head(20))
        print("="*80 + "\n")
        
    if not out_forecasts.empty:
        out_forecasts.to_parquet(settings.paths.processed_forecasts, index=False)
        logger.info(f"Saved forecasts to {settings.paths.processed_forecasts}")
