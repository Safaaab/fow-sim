"""Data Explorer page for FoW-Sim dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from fowsim.config.settings import Settings


st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("Data Explorer")
st.markdown("Explore historical workforce and technology trends across countries.")

settings = Settings()

# Load data
@st.cache_data
def load_panel_data():
    panel_path = settings.paths.processed_panel
    if not panel_path.exists():
        return None
    return pd.read_parquet(panel_path)

panel = load_panel_data()

if panel is None:
    st.error("No dataset found. Run `python -m fowsim.cli build-data` first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
countries = sorted(panel["iso3"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries",
    countries,
    default=countries[:5] if len(countries) >= 5 else countries
)

# Get numeric columns for indicators
numeric_cols = [c for c in panel.columns if c not in {"iso3", "year", "country"} 
                and pd.api.types.is_numeric_dtype(panel[c])]

# Exclude lag/rolling features for cleaner view
base_indicators = [c for c in numeric_cols if not any(x in c for x in ["_lag", "_roll"])]

selected_indicator = st.sidebar.selectbox(
    "Indicator",
    base_indicators,
    index=0 if base_indicators else None
)

if not selected_indicator or not selected_countries:
    st.warning("Please select at least one country and indicator.")
    st.stop()

# Filter data
filtered_data = panel[panel["iso3"].isin(selected_countries)].copy()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Time Series: {selected_indicator}")
    
    # Create line chart
    fig = px.line(
        filtered_data,
        x="year",
        y=selected_indicator,
        color="iso3",
        title=f"{selected_indicator} Over Time",
        labels={"iso3": "Country", "year": "Year"},
        markers=True
    )
    fig.update_layout(hovermode="x unified", height=500)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Statistics")
    
    # Summary statistics
    stats = filtered_data.groupby("iso3")[selected_indicator].agg([
        ("Mean", "mean"),
        ("Std Dev", "std"),
        ("Min", "min"),
        ("Max", "max"),
        ("Latest", "last")
    ]).round(2)
    
    st.dataframe(stats, width='stretch')

# Country comparison
st.subheader("Country Comparison")

col1, col2 = st.columns(2)

with col1:
    # Latest year comparison
    latest_year = filtered_data["year"].max()
    latest_data = filtered_data[filtered_data["year"] == latest_year]
    
    fig_bar = px.bar(
        latest_data.sort_values(selected_indicator, ascending=False),
        x="iso3",
        y=selected_indicator,
        title=f"{selected_indicator} - {int(latest_year)}",
        labels={"iso3": "Country"},
        color=selected_indicator,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_bar, width='stretch')

with col2:
    # Growth rate analysis
    growth_data = []
    for country in selected_countries:
        country_data = filtered_data[filtered_data["iso3"] == country].sort_values("year")
        if len(country_data) >= 2:
            first_val = country_data[selected_indicator].iloc[0]
            last_val = country_data[selected_indicator].iloc[-1]
            if pd.notna(first_val) and pd.notna(last_val) and first_val != 0:
                growth_rate = ((last_val - first_val) / abs(first_val)) * 100
                growth_data.append({"Country": country, "Growth (%)": growth_rate})
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        fig_growth = px.bar(
            growth_df.sort_values("Growth (%)", ascending=False),
            x="Country",
            y="Growth (%)",
            title=f"{selected_indicator} - Total Growth %",
            color="Growth (%)",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_growth, width='stretch')

# Correlation analysis
st.subheader("Correlation Analysis")

# Select multiple indicators for correlation
selected_indicators_corr = st.multiselect(
    "Select indicators for correlation analysis",
    base_indicators,
    default=base_indicators[:5] if len(base_indicators) >= 5 else base_indicators
)

if len(selected_indicators_corr) >= 2:
    # Calculate correlation matrix
    corr_data = filtered_data[selected_indicators_corr].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale="RdBu",
        zmid=0,
        text=corr_data.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig_corr.update_layout(
        title="Correlation Matrix",
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    st.plotly_chart(fig_corr, width='stretch')

# Data table
st.subheader("Raw Data")
st.dataframe(
    filtered_data[["iso3", "year"] + [selected_indicator]].sort_values(["iso3", "year"], ascending=[True, False]),
    width='stretch',
    height=300
)

# Download option
csv = filtered_data.to_csv(index=False)
st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name=f"fow_sim_data_{selected_indicator}.csv",
    mime="text/csv"
)
