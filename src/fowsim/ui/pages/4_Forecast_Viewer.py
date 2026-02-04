"""Forecast Viewer page for FoW-Sim dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from fowsim.config.settings import Settings


st.set_page_config(page_title="Forecast Viewer", page_icon="📊", layout="wide")

st.title("Forecast Viewer")
st.markdown("Explore detailed model forecasts with confidence intervals.")

settings = Settings()

# Load data
@st.cache_data
def load_forecasts():
    forecast_path = settings.paths.processed_forecasts
    if not forecast_path.exists():
        return None
    return pd.read_parquet(forecast_path)

forecasts = load_forecasts()

if forecasts is None:
    st.error("No forecasts found. Run `python -m fowsim.cli train` first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

targets = sorted(forecasts["target"].unique().tolist())
selected_target = st.sidebar.selectbox("Target Variable", targets, index=0)

horizons = sorted(forecasts["horizon"].unique().tolist())
selected_horizon = st.sidebar.selectbox("Forecast Horizon (years)", horizons, index=0)

countries = sorted(forecasts["iso3"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries",
    countries,
    default=countries[:5] if len(countries) >= 5 else countries
)

if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()

# Filter data
filtered_forecasts = forecasts[
    (forecasts["target"] == selected_target) &
    (forecasts["horizon"] == selected_horizon) &
    (forecasts["iso3"].isin(selected_countries))
]

# Summary metrics
st.subheader(f"Summary: {selected_target} ({selected_horizon}-year horizon)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_forecasts = len(filtered_forecasts)
    st.metric("Total Forecasts", total_forecasts)

with col2:
    if "model" in filtered_forecasts.columns:
        model_name = filtered_forecasts["model"].iloc[0]
        st.metric("Model Used", model_name)

with col3:
    avg_pred = filtered_forecasts["y_pred"].mean()
    st.metric("Avg Prediction", f"{avg_pred:.2f}")

with col4:
    avg_true = filtered_forecasts["y_true"].mean()
    st.metric("Avg Actual", f"{avg_true:.2f}")

# Forecast vs Actual visualization
st.subheader("Forecast vs Actual Values")

# Create comparison chart for each country
for country in selected_countries:
    country_data = filtered_forecasts[filtered_forecasts["iso3"] == country].sort_values("year")
    
    if len(country_data) == 0:
        continue
    
    with st.expander(f"{country}", expanded=(len(selected_countries) <= 3)):
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=country_data["year"],
            y=country_data["y_true"],
            name="Actual",
            mode="lines+markers",
            line=dict(color="blue"),
            marker=dict(size=6)
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=country_data["year"],
            y=country_data["y_pred"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="red", dash="dash"),
            marker=dict(size=6)
        ))
        
        # Confidence intervals (if available)
        if "lower_ci" in country_data.columns and "upper_ci" in country_data.columns:
            fig.add_trace(go.Scatter(
                x=country_data["year"].tolist() + country_data["year"].tolist()[::-1],
                y=country_data["upper_ci"].tolist() + country_data["lower_ci"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"{selected_target} - {country}",
            xaxis_title="Year",
            yaxis_title=selected_target,
            hovermode="x unified",
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Accuracy metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        mae = mean_absolute_error(country_data["y_true"], country_data["y_pred"])
        rmse = np.sqrt(mean_squared_error(country_data["y_true"], country_data["y_pred"]))
        r2 = r2_score(country_data["y_true"], country_data["y_pred"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.3f}")
        col2.metric("RMSE", f"{rmse:.3f}")
        col3.metric("R² Score", f"{r2:.3f}")

# Scatter plot: Predicted vs Actual
st.subheader("Prediction Accuracy Scatter Plot")

fig_scatter = px.scatter(
    filtered_forecasts,
    x="y_true",
    y="y_pred",
    color="iso3",
    title="Predicted vs Actual Values (All Countries)",
    labels={"y_true": "Actual Value", "y_pred": "Predicted Value", "iso3": "Country"},
    trendline="ols",
    hover_data=["year"]
)

# Add perfect prediction line
min_val = min(filtered_forecasts["y_true"].min(), filtered_forecasts["y_pred"].min())
max_val = max(filtered_forecasts["y_true"].max(), filtered_forecasts["y_pred"].max())

fig_scatter.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Prediction",
        line=dict(dash="dash", color="gray")
    )
)

fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, width='stretch')

# Residual analysis
st.subheader("Residual Analysis")

filtered_forecasts = filtered_forecasts.copy()  # Avoid SettingWithCopyWarning
filtered_forecasts.loc[:, "residual"] = filtered_forecasts["y_true"] - filtered_forecasts["y_pred"]

col1, col2 = st.columns(2)

with col1:
    # Residuals over time
    fig_residual_time = px.scatter(
        filtered_forecasts,
        x="year",
        y="residual",
        color="iso3",
        title="Residuals Over Time",
        labels={"year": "Year", "residual": "Residual (Actual - Predicted)", "iso3": "Country"}
    )
    fig_residual_time.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_residual_time, width='stretch')

with col2:
    # Residual distribution
    fig_residual_hist = px.histogram(
        filtered_forecasts,
        x="residual",
        nbins=30,
        title="Residual Distribution",
        labels={"residual": "Residual (Actual - Predicted)"},
        color_discrete_sequence=["#636EFA"]
    )
    fig_residual_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residual_hist, width='stretch')

# Comparison table
st.subheader("Detailed Forecast Table")

display_cols = ["iso3", "year", "y_true", "y_pred", "residual"]
if "model" in filtered_forecasts.columns:
    display_cols.append("model")
if "lower_ci" in filtered_forecasts.columns and "upper_ci" in filtered_forecasts.columns:
    display_cols.extend(["lower_ci", "upper_ci"])

display_df = filtered_forecasts[display_cols].sort_values(["iso3", "year"])
display_df = display_df.round(3)

st.dataframe(
    display_df,
    width='stretch',
    height=400
)

# Export options
st.subheader("Export Forecasts")

csv = filtered_forecasts.to_csv(index=False)
st.download_button(
    label="Download Forecast Data",
    data=csv,
    file_name=f"forecasts_{selected_target}_{selected_horizon}y.csv",
    mime="text/csv"
)
