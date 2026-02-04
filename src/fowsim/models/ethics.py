"""Ethics and fairness evaluation module for Future of Work predictions.

Implements IEEE AI Ethics Framework principles:
- Transparency
- Accountability
- Fairness
- Privacy
- Reliability
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class FairnessMetrics:
    """Container for fairness evaluation metrics."""
    demographic_parity_diff: float
    equal_opportunity_diff: float
    disparate_impact: float
    statistical_parity: float
    group_metrics: Dict[str, Dict[str, float]]


@dataclass
class BiasReport:
    """Comprehensive bias assessment report."""
    data_bias: Dict[str, Any]
    prediction_bias: Dict[str, Any]
    fairness_metrics: FairnessMetrics
    recommendations: List[str]


def calculate_demographic_parity(
    predictions: pd.Series,
    protected_attribute: pd.Series,
    threshold: float = 0.5
) -> float:
    """Calculate demographic parity difference.
    
    Measures if positive prediction rate is similar across groups.
    Target: close to 0 (perfectly fair)
    """
    groups = protected_attribute.unique()
    if len(groups) < 2:
        return 0.0
    
    positive_rates = []
    for group in groups:
        mask = protected_attribute == group
        if mask.sum() > 0:
            rate = (predictions[mask] > threshold).mean()
            positive_rates.append(rate)
    
    if len(positive_rates) < 2:
        return 0.0
    
    # Return max difference between groups
    return float(max(positive_rates) - min(positive_rates))


def calculate_equal_opportunity(
    predictions: pd.Series,
    actuals: pd.Series,
    protected_attribute: pd.Series,
    threshold: float = 0.5
) -> float:
    """Calculate equal opportunity difference.
    
    Measures if true positive rate is similar across groups.
    Target: close to 0 (perfectly fair)
    """
    groups = protected_attribute.unique()
    if len(groups) < 2:
        return 0.0
    
    tpr_values = []
    for group in groups:
        mask = protected_attribute == group
        positives = actuals[mask] > threshold
        
        if positives.sum() > 0:
            tp = ((predictions[mask] > threshold) & positives).sum()
            tpr = tp / positives.sum()
            tpr_values.append(tpr)
    
    if len(tpr_values) < 2:
        return 0.0
    
    return float(max(tpr_values) - min(tpr_values))


def calculate_disparate_impact(
    predictions: pd.Series,
    protected_attribute: pd.Series,
    threshold: float = 0.5
) -> float:
    """Calculate disparate impact ratio.
    
    Ratio of positive rate for unprivileged to privileged group.
    Target: close to 1.0 (perfectly fair)
    Range: [0, inf], acceptable range: [0.8, 1.25]
    """
    groups = protected_attribute.unique()
    if len(groups) < 2:
        return 1.0
    
    rates = {}
    for group in groups:
        mask = protected_attribute == group
        if mask.sum() > 0:
            rate = (predictions[mask] > threshold).mean()
            rates[group] = rate
    
    if len(rates) < 2:
        return 1.0
    
    # Assume first group is unprivileged (can be configured)
    unprivileged_rate = list(rates.values())[0]
    privileged_rate = max(rates.values())
    
    if privileged_rate == 0:
        return 1.0
    
    return float(unprivileged_rate / privileged_rate)


def analyze_data_bias(panel: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset for potential biases.
    
    Checks:
    - Missing data patterns by country
    - Temporal coverage gaps
    - Representation bias (sample size)
    - Feature correlation with protected attributes
    """
    bias_report = {}
    
    # Missing data analysis
    missing_by_country = panel.groupby("iso3").apply(
        lambda df: (df.isnull().sum() / len(df) * 100).mean()
    ).sort_values(ascending=False)
    
    bias_report["missing_data_top_countries"] = missing_by_country.head(5).to_dict()
    bias_report["missing_data_avg"] = float(missing_by_country.mean())
    
    # Temporal coverage
    years_by_country = panel.groupby("iso3")["year"].apply(lambda x: len(x.unique()))
    bias_report["temporal_coverage_min"] = int(years_by_country.min())
    bias_report["temporal_coverage_max"] = int(years_by_country.max())
    bias_report["temporal_coverage_std"] = float(years_by_country.std())
    
    # Sample size distribution
    samples_by_country = panel.groupby("iso3").size()
    bias_report["sample_size_min"] = int(samples_by_country.min())
    bias_report["sample_size_max"] = int(samples_by_country.max())
    bias_report["sample_size_cv"] = float(samples_by_country.std() / samples_by_country.mean())
    
    # Check for underrepresented groups
    underrepresented = samples_by_country[samples_by_country < samples_by_country.quantile(0.25)]
    bias_report["underrepresented_countries"] = underrepresented.to_dict()
    
    return bias_report


def analyze_prediction_bias(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame
) -> Dict[str, Any]:
    """Analyze predictions for systematic bias.
    
    Checks:
    - Prediction error distribution across countries
    - Over/under-prediction patterns
    - Confidence interval coverage
    """
    bias_report = {}
    
    # Calculate residuals
    forecasts["residual"] = forecasts["y_true"] - forecasts["y_pred"]
    
    # Error distribution by country
    error_by_country = forecasts.groupby("iso3")["residual"].agg([
        ("mean_error", "mean"),
        ("std_error", "std"),
        ("abs_error", lambda x: x.abs().mean())
    ])
    
    # Identify countries with systematic over/under-prediction
    systematic_bias = error_by_country[abs(error_by_country["mean_error"]) > error_by_country["std_error"]]
    bias_report["systematic_bias_countries"] = systematic_bias["mean_error"].to_dict()
    
    # Check if errors correlate with country characteristics
    # (e.g., GDP, development level)
    if "gdp_per_capita" in panel.columns:
        panel_latest = panel.loc[panel.groupby("iso3")["year"].idxmax()]
        merged = forecasts.merge(
            panel_latest[["iso3", "gdp_per_capita"]],
            on="iso3",
            how="left"
        )
        
        if "gdp_per_capita" in merged.columns:
            correlation = merged[["residual", "gdp_per_capita"]].corr().iloc[0, 1]
            bias_report["error_gdp_correlation"] = float(correlation)
    
    # Confidence interval coverage (if available)
    if "lower_ci" in forecasts.columns and "upper_ci" in forecasts.columns:
        coverage = (
            (forecasts["y_true"] >= forecasts["lower_ci"]) &
            (forecasts["y_true"] <= forecasts["upper_ci"])
        ).mean()
        bias_report["ci_coverage"] = float(coverage)
        bias_report["ci_target"] = 0.95
    
    return bias_report


def evaluate_fairness(
    forecasts: pd.DataFrame,
    protected_attribute: str = "iso3"
) -> FairnessMetrics:
    """Evaluate prediction fairness across groups.
    
    Args:
        forecasts: DataFrame with predictions and actuals
        protected_attribute: Column name for group membership (e.g., country)
    
    Returns:
        FairnessMetrics object with fairness scores
    """
    # Group-level metrics
    group_metrics = {}
    for group in forecasts[protected_attribute].unique():
        group_data = forecasts[forecasts[protected_attribute] == group]
        
        mae = (group_data["y_true"] - group_data["y_pred"]).abs().mean()
        rmse = np.sqrt(((group_data["y_true"] - group_data["y_pred"]) ** 2).mean())
        
        group_metrics[str(group)] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "count": len(group_data)
        }
    
    # Calculate fairness metrics
    dp_diff = calculate_demographic_parity(
        forecasts["y_pred"],
        forecasts[protected_attribute]
    )
    
    eo_diff = calculate_equal_opportunity(
        forecasts["y_pred"],
        forecasts["y_true"],
        forecasts[protected_attribute]
    )
    
    di = calculate_disparate_impact(
        forecasts["y_pred"],
        forecasts[protected_attribute]
    )
    
    # Statistical parity: variance in prediction rates
    pred_rates = forecasts.groupby(protected_attribute)["y_pred"].mean()
    stat_parity = float(pred_rates.std() / pred_rates.mean()) if pred_rates.mean() != 0 else 0.0
    
    return FairnessMetrics(
        demographic_parity_diff=dp_diff,
        equal_opportunity_diff=eo_diff,
        disparate_impact=di,
        statistical_parity=stat_parity,
        group_metrics=group_metrics
    )


def generate_bias_report(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame
) -> BiasReport:
    """Generate comprehensive bias and fairness report.
    
    Follows IEEE AI Ethics Framework principles.
    """
    # Analyze data bias
    data_bias = analyze_data_bias(panel)
    
    # Analyze prediction bias
    prediction_bias = analyze_prediction_bias(forecasts, panel)
    
    # Evaluate fairness
    fairness_metrics = evaluate_fairness(forecasts)
    
    # Generate recommendations
    recommendations = []
    
    # Data bias recommendations
    if data_bias["missing_data_avg"] > 20:
        recommendations.append(
            "High average missing data (>20%). Consider data imputation strategies."
        )
    
    if data_bias["sample_size_cv"] > 0.5:
        recommendations.append(
            "High variability in sample sizes across countries. "
            "Consider stratified sampling or weighting."
        )
    
    if len(data_bias["underrepresented_countries"]) > 0:
        recommendations.append(
            f"{len(data_bias['underrepresented_countries'])} underrepresented countries detected. "
            "Consider collecting more data or using transfer learning."
        )
    
    # Prediction bias recommendations
    if len(prediction_bias.get("systematic_bias_countries", {})) > 0:
        recommendations.append(
            f"Systematic bias detected in {len(prediction_bias['systematic_bias_countries'])} countries. "
            "Model may need country-specific calibration."
        )
    
    if "error_gdp_correlation" in prediction_bias:
        corr = prediction_bias["error_gdp_correlation"]
        if abs(corr) > 0.3:
            recommendations.append(
                f"Prediction errors correlate with GDP (r={corr:.2f}). "
                "Model may disadvantage low/high-income countries."
            )
    
    if "ci_coverage" in prediction_bias:
        coverage = prediction_bias["ci_coverage"]
        if coverage < 0.90:
            recommendations.append(
                f"Confidence interval coverage low ({coverage:.1%} < 95%). "
                "Uncertainty estimates may be unreliable."
            )
    
    # Fairness recommendations
    if fairness_metrics.disparate_impact < 0.8 or fairness_metrics.disparate_impact > 1.25:
        recommendations.append(
            f"Disparate impact ratio ({fairness_metrics.disparate_impact:.2f}) "
            "outside acceptable range [0.8, 1.25]. Review for algorithmic fairness."
        )
    
    if fairness_metrics.demographic_parity_diff > 0.1:
        recommendations.append(
            f"Demographic parity difference high ({fairness_metrics.demographic_parity_diff:.2f}). "
            "Predictions may be imbalanced across groups."
        )
    
    if not recommendations:
        recommendations.append("No major fairness or bias concerns detected.")
    
    return BiasReport(
        data_bias=data_bias,
        prediction_bias=prediction_bias,
        fairness_metrics=fairness_metrics,
        recommendations=recommendations
    )


def print_ethics_report(report: BiasReport) -> None:
    """Print formatted ethics and fairness report."""
    print("\n" + "="*80)
    print("ETHICS & FAIRNESS EVALUATION REPORT")
    print("="*80)
    
    print("\nDATA BIAS ANALYSIS")
    print("-" * 80)
    print(f"Average Missing Data: {report.data_bias['missing_data_avg']:.1f}%")
    print(f"Temporal Coverage Range: {report.data_bias['temporal_coverage_min']}-"
          f"{report.data_bias['temporal_coverage_max']} years")
    print(f"Sample Size CV: {report.data_bias['sample_size_cv']:.2f}")
    print(f"Underrepresented Countries: {len(report.data_bias['underrepresented_countries'])}")
    
    print("\nPREDICTION BIAS ANALYSIS")
    print("-" * 80)
    if "systematic_bias_countries" in report.prediction_bias:
        print(f"Countries with Systematic Bias: {len(report.prediction_bias['systematic_bias_countries'])}")
    if "error_gdp_correlation" in report.prediction_bias:
        print(f"Error-GDP Correlation: {report.prediction_bias['error_gdp_correlation']:.3f}")
    if "ci_coverage" in report.prediction_bias:
        print(f"CI Coverage: {report.prediction_bias['ci_coverage']:.1%} (target: 95%)")
    
    print("\nFAIRNESS METRICS")
    print("-" * 80)
    print(f"Demographic Parity Diff: {report.fairness_metrics.demographic_parity_diff:.3f}")
    print(f"Equal Opportunity Diff: {report.fairness_metrics.equal_opportunity_diff:.3f}")
    print(f"Disparate Impact: {report.fairness_metrics.disparate_impact:.3f} (target: ~1.0)")
    print(f"Statistical Parity: {report.fairness_metrics.statistical_parity:.3f}")
    
    print("\nRECOMMENDATIONS")
    print("-" * 80)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n" + "="*80 + "\n")
