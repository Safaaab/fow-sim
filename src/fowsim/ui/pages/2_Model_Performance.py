"""Model Performance page for FoW-Sim dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from fowsim.config.settings import Settings


st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")

st.title("Model Performance")
st.markdown("Analyze forecasting model accuracy and backtest results.")

settings = Settings()

# Load backtest metrics
@st.cache_data
def load_metrics():
    metrics_path = settings.paths.backtest_metrics
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)

metrics = load_metrics()

if metrics is None:
    st.error("No model metrics found. Run `python -m fowsim.cli train` first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

targets = sorted(metrics["target"].unique().tolist())
selected_target = st.sidebar.selectbox("Target Variable", targets, index=0)

horizons = sorted(metrics["horizon"].unique().tolist())
selected_horizon = st.sidebar.selectbox("Forecast Horizon (years)", horizons, index=0)

# Filter data
filtered_metrics = metrics[
    (metrics["target"] == selected_target) & 
    (metrics["horizon"] == selected_horizon)
]

# Model comparison
st.subheader(f"Model Comparison: {selected_target} ({selected_horizon}-year horizon)")

col1, col2, col3 = st.columns(3)

# Aggregate metrics by model
model_summary = filtered_metrics.groupby("model").agg({
    "rmse": ["mean", "std"],
    "mae": ["mean", "std"]
}).round(3)

model_summary.columns = ["RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std"]
model_summary = model_summary.reset_index().sort_values("RMSE_mean")

with col1:
    st.metric(
        "Best Model",
        model_summary.iloc[0]["model"],
        f"RMSE: {model_summary.iloc[0]['RMSE_mean']:.3f}"
    )

with col2:
    avg_rmse = model_summary["RMSE_mean"].mean()
    st.metric(
        "Average RMSE",
        f"{avg_rmse:.3f}",
        f"±{model_summary['RMSE_std'].mean():.3f}"
    )

with col3:
    avg_mae = model_summary["MAE_mean"].mean()
    st.metric(
        "Average MAE",
        f"{avg_mae:.3f}",
        f"±{model_summary['MAE_std'].mean():.3f}"
    )

# Model performance charts
col1, col2 = st.columns(2)

with col1:
    # RMSE comparison
    fig_rmse = px.bar(
        model_summary,
        x="model",
        y="RMSE_mean",
        error_y="RMSE_std",
        title="RMSE by Model",
        labels={"model": "Model", "RMSE_mean": "RMSE"},
        color="RMSE_mean",
        color_continuous_scale="Reds_r"
    )
    fig_rmse.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_rmse, width='stretch')

with col2:
    # MAE comparison
    fig_mae = px.bar(
        model_summary,
        x="model",
        y="MAE_mean",
        error_y="MAE_std",
        title="MAE by Model",
        labels={"model": "Model", "MAE_mean": "MAE"},
        color="MAE_mean",
        color_continuous_scale="Blues_r"
    )
    fig_mae.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_mae, width='stretch')

# Time series of model performance
st.subheader("Performance Over Time")

# Select model to analyze
selected_model = st.selectbox(
    "Select model for detailed analysis",
    model_summary["model"].tolist(),
    index=0
)

model_data = filtered_metrics[filtered_metrics["model"] == selected_model]

col1, col2 = st.columns(2)

with col1:
    fig_rmse_time = px.line(
        model_data,
        x="test_year",
        y="rmse",
        title=f"RMSE Over Test Years - {selected_model}",
        markers=True,
        labels={"test_year": "Test Year", "rmse": "RMSE"}
    )
    st.plotly_chart(fig_rmse_time, width='stretch')

with col2:
    fig_mae_time = px.line(
        model_data,
        x="test_year",
        y="mae",
        title=f"MAE Over Test Years - {selected_model}",
        markers=True,
        labels={"test_year": "Test Year", "mae": "MAE"},
        color_discrete_sequence=["#EF553B"]
    )
    st.plotly_chart(fig_mae_time, width='stretch')

# Scatter plot: RMSE vs MAE
st.subheader("RMSE vs MAE Comparison")

fig_scatter = px.scatter(
    filtered_metrics,
    x="rmse",
    y="mae",
    color="model",
    size="n_test" if "n_test" in filtered_metrics.columns else None,
    hover_data=["test_year", "train_end_year"],
    title="RMSE vs MAE by Model",
    labels={"rmse": "RMSE", "mae": "MAE"}
)
fig_scatter.add_trace(
    go.Scatter(
        x=[0, filtered_metrics["rmse"].max()],
        y=[0, filtered_metrics["rmse"].max()],
        mode="lines",
        name="y=x",
        line=dict(dash="dash", color="gray")
    )
)
st.plotly_chart(fig_scatter, width='stretch')

# All targets and horizons summary
st.subheader("Full Model Comparison (All Targets & Horizons)")

full_summary = metrics.groupby(["target", "horizon", "model"]).agg({
    "rmse": "mean",
    "mae": "mean"
}).round(3).reset_index()

# Find best model for each target-horizon combination
best_models = full_summary.loc[full_summary.groupby(["target", "horizon"])["rmse"].idxmin()]

st.dataframe(
    best_models.style.background_gradient(subset=["rmse", "mae"], cmap="RdYlGn_r"),
    width='stretch'
)

# Horizon comparison
st.subheader("Forecast Horizon Analysis")

horizon_comparison = metrics.groupby(["horizon", "model"]).agg({
    "rmse": "mean",
    "mae": "mean"
}).reset_index()

fig_horizon = px.line(
    horizon_comparison,
    x="horizon",
    y="rmse",
    color="model",
    title="Model Performance by Forecast Horizon",
    markers=True,
    labels={"horizon": "Forecast Horizon (years)", "rmse": "Average RMSE"}
)
st.plotly_chart(fig_horizon, width='stretch')

# Download metrics
st.subheader("Export Results")
csv = filtered_metrics.to_csv(index=False)
st.download_button(
    label="Download Metrics",
    data=csv,
    file_name=f"model_metrics_{selected_target}_{selected_horizon}y.csv",
    mime="text/csv"
)
