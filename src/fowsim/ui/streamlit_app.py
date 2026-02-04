"""Main Streamlit dashboard for Future of Work Simulator."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from fowsim.config.settings import Settings
from fowsim.simulation.scenarios import scenario_registry


st.set_page_config(
    page_title="FoW-Sim Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

settings = Settings()
scens = scenario_registry()

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("FoW-Sim — Future of Work Simulator")
st.markdown("""
**Predictive Analysis and Simulation of the Future of Work Using AI and Machine Learning**

This dashboard provides:
- Historical workforce and technology trend analysis
- AI/ML-powered forecasting models (5, 10, 20-year horizons)
- Interactive scenario simulations
- Model performance evaluation and comparison
""")

# Load data
@st.cache_data
def load_panel_data():
    panel_path = settings.paths.processed_panel
    if not panel_path.exists():
        return None
    return pd.read_parquet(panel_path)

@st.cache_data
def load_forecasts():
    forecast_path = settings.paths.processed_forecasts
    if not forecast_path.exists():
        return None
    return pd.read_parquet(forecast_path)

@st.cache_data
def load_metrics():
    metrics_path = settings.paths.backtest_metrics
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)

panel = load_panel_data()
forecasts = load_forecasts()
metrics = load_metrics()

if panel is None:
    st.error("No dataset found. Please run: `python -m fowsim.cli build-data`")
    st.info("""
    **Getting Started:**
    1. Open terminal in project directory
    2. Run: `python -m fowsim.cli build-data --start-year 2000 --end-year 2024`
    3. Run: `python -m fowsim.cli train --horizons 5 10 20`
    4. Refresh this page
    """)
    st.stop()

# Sidebar
st.sidebar.header("Quick Navigator")
st.sidebar.markdown("""
**[Data Explorer](Data_Explorer)** - Explore historical trends

**[Model Performance](Model_Performance)** - Evaluate forecasting accuracy

**[Scenario Analysis](Scenario_Analysis)** - Compare future scenarios

**[Forecast Viewer](Forecast_Viewer)** - View model predictions
""")

st.sidebar.markdown("---")
st.sidebar.header("Dataset Info")

# Dataset statistics
countries = sorted(panel["iso3"].dropna().unique().tolist())
years = sorted(panel["year"].dropna().unique().tolist())
indicators = len([c for c in panel.columns if c not in {"iso3", "year", "country"}])

st.sidebar.metric("Countries", len(countries))
st.sidebar.metric("Years", f"{int(min(years))}-{int(max(years))}")
st.sidebar.metric("Indicators", indicators)
st.sidebar.metric("Total Observations", len(panel))

if metrics is not None:
    st.sidebar.metric("Trained Models", metrics["model"].nunique())

st.sidebar.markdown("---")
st.sidebar.info("""
**Project**: BSc (Hons) Computer Science  
**Module**: CMP6200  
**Year**: 2025/26
""")

# Main Dashboard Content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Countries Analyzed",
        len(countries),
        help="Number of countries in the dataset"
    )

with col2:
    st.metric(
        "Time Span",
        f"{max(years) - min(years)} years",
        help="Historical data range"
    )

with col3:
    st.metric(
        "Forecast Horizons",
        "5, 10, 20 years",
        help="Available forecast periods"
    )

with col4:
    st.metric(
        "Scenarios",
        len(scens),
        help="Number of future scenarios"
    )

st.markdown("---")

# Quick Visualization Section
st.subheader("Quick Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Countries by Latest Data")
    
    # Show top countries by a key indicator
    latest_year = panel["year"].max()
    latest_data = panel[panel["year"] == latest_year]
    
    if "gdp_per_capita" in latest_data.columns:
        top_countries = latest_data.nlargest(10, "gdp_per_capita")[["iso3", "gdp_per_capita"]]
        
        fig_top = px.bar(
            top_countries,
            x="gdp_per_capita",
            y="iso3",
            orientation="h",
            title=f"Top 10 Countries by GDP per Capita ({int(latest_year)})",
            labels={"gdp_per_capita": "GDP per Capita", "iso3": "Country"},
            color="gdp_per_capita",
            color_continuous_scale="Viridis"
        )
        fig_top.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_top, width='stretch')

with col2:
    st.markdown("### Technology Adoption Trends")
    
    if "internet_users" in panel.columns:
        # Calculate average internet users by year
        internet_trend = panel.groupby("year")["internet_users"].mean().reset_index()
        
        fig_internet = px.line(
            internet_trend,
            x="year",
            y="internet_users",
            title="Global Average Internet Users (%)",
            markers=True,
            labels={"year": "Year", "internet_users": "Internet Users (%)"}
        )
        fig_internet.update_layout(height=400)
        st.plotly_chart(fig_internet, width='stretch')

# Model Performance Summary (if available)
if metrics is not None:
    st.markdown("---")
    st.subheader("Model Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_rmse = metrics["rmse"].mean()
        st.metric("Average RMSE", f"{avg_rmse:.3f}")
    
    with col2:
        avg_mae = metrics["mae"].mean()
        st.metric("Average MAE", f"{avg_mae:.3f}")
    
    with col3:
        best_model = metrics.groupby("model")["rmse"].mean().idxmin()
        st.metric("Best Overall Model", best_model)
    
    # Model comparison chart
    model_comparison = metrics.groupby("model").agg({"rmse": "mean", "mae": "mean"}).reset_index()
    model_comparison = model_comparison.sort_values("rmse")
    
    fig_models = px.bar(
        model_comparison,
        x="model",
        y="rmse",
        title="Average RMSE by Model (All Targets & Horizons)",
        labels={"model": "Model", "rmse": "RMSE"},
        color="rmse",
        color_continuous_scale="Reds_r"
    )
    fig_models.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_models, width='stretch')

# Scenario Previews
st.markdown("---")
st.subheader("Available Scenarios")

scenario_cols = st.columns(3)
scenario_list = list(scens.values())

for idx, scenario in enumerate(scenario_list):
    col_idx = idx % 3
    with scenario_cols[col_idx]:
        with st.container():
            st.markdown(f"**{scenario.name}**")
            st.caption(scenario.description[:100] + "..." if len(scenario.description) > 100 else scenario.description)

# Getting Started Guide
st.markdown("---")
st.subheader("Getting Started")

with st.expander("How to use this dashboard"):
    st.markdown("""
    ### Navigation
    Use the sidebar or links above to navigate between different analysis pages:
    
    1. **Data Explorer**: Browse and visualize historical data
        - Compare countries and indicators
        - View correlations and trends
        - Export filtered datasets
    
    2. **Model Performance**: Evaluate forecasting accuracy
        - Compare different ML models
        - Analyze backtest results
        - View performance metrics
    
    3. **Scenario Analysis**: Simulate future scenarios
        - Compare multiple scenarios
        - Visualize 5/10/20-year projections
        - Understand scenario impacts
    
    4. **Forecast Viewer**: Examine detailed forecasts
        - View predictions with confidence intervals
        - Compare across countries and horizons
    
    ### CLI Commands
    ```bash
    # Build dataset
    python -m fowsim.cli build-data --start-year 1990 --end-year 2024
    
    # Train models
    python -m fowsim.cli train --horizons 5 10 20
    
    # Run simulation
    python -m fowsim.cli simulate --country GBR --scenario rapid_ai --horizon 10
    ```
    """)

with st.expander("About this project"):
    st.markdown("""
    ### Project Overview
    **Title**: Predictive Analysis and Simulation of the Future of Work Using AI and Machine Learning
    
    **Objectives**:
    - Analyze historical workforce and technology trends
    - Develop AI/ML predictive models
    - Visualize 5, 10, and 20-year future scenarios
    - Evaluate model performance and accuracy
    - Assess ethical and sustainability implications
    
    **Technologies**:
    - Python, Pandas, NumPy, Scikit-learn
    - XGBoost, LightGBM (optional)
    - Statsmodels (ARIMA/SARIMA)
    - Plotly, Streamlit
    - IBM Watson NLU (optional)
    
    **Data Sources**:
    - World Bank Open Data
    - OECD Statistics
    - Kaggle datasets (automation risk, remote work trends)
    
    **Ethical Compliance**:
    - BCU Ethical Application #13740
    - IEEE AI Ethics Framework
    - Anonymized public data only
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Future of Work Simulator</b> | Developed for BSc (Hons) Computer Science | CMP6200</p>
    <p>© 2025/26 | Birmingham City University</p>
</div>
""", unsafe_allow_html=True)
