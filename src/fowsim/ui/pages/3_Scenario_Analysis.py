"""Scenario Comparison page for FoW-Sim dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from fowsim.config.settings import Settings
from fowsim.simulation.scenarios import scenario_registry, get_scenario_categories
from fowsim.simulation.simulator import _make_future_frame


st.set_page_config(page_title="Scenario Analysis", page_icon="📊", layout="wide")

st.title("Scenario Analysis")
st.markdown("Compare different future of work scenarios (5, 10, 20-year projections).")

settings = Settings()
scenarios = scenario_registry()

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

# Sidebar configuration
st.sidebar.header("Configuration")

countries = sorted(panel["iso3"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries, index=0 if countries else None)

horizons = [5, 10, 20]
selected_horizon = st.sidebar.selectbox("Forecast Horizon (years)", horizons, index=2)

# Scenario selection
scenario_categories = get_scenario_categories()
st.sidebar.subheader("Select Scenarios to Compare")

selected_scenarios = []
checkbox_counter = 0  # Ensure unique keys even if scenario names repeat
for category, scenario_names in scenario_categories.items():
    with st.sidebar.expander(category):
        for name in scenario_names:
            if name in scenarios:
                checkbox_counter += 1
                if st.checkbox(scenarios[name].description[:50] + "...", value=(name == "baseline"), key=f"scenario_{name}_{checkbox_counter}"):
                    selected_scenarios.append(name)

if not selected_scenarios:
    st.warning("Please select at least one scenario to analyze.")
    st.stop()

# Get base indicators (exclude lag/rolling features)
numeric_cols = [c for c in panel.columns if c not in {"iso3", "year", "country"} 
                and pd.api.types.is_numeric_dtype(panel[c])]
base_indicators = [c for c in numeric_cols if not any(x in c for x in ["_lag", "_roll"])]

selected_indicator = st.selectbox("Indicator to Analyze", base_indicators, index=0)

# Generate scenario data
st.subheader(f"Scenario Projections: {selected_country} - {selected_horizon} years")

@st.cache_data
def generate_scenario_projections(country, horizon, scenario_names):
    """Generate projections for all selected scenarios."""
    results = {}
    
    for scenario_name in scenario_names:
        try:
            future = _make_future_frame(panel, country=country, horizon=horizon)
            future_modified = scenarios[scenario_name].apply(future)
            results[scenario_name] = future_modified
        except Exception as e:
            st.warning(f"Could not generate {scenario_name}: {e}")
    
    return results

scenario_data = generate_scenario_projections(selected_country, selected_horizon, selected_scenarios)

if not scenario_data:
    st.error("Failed to generate scenario projections.")
    st.stop()

# Prepare comparison data
comparison_data = []
for scenario_name, future_df in scenario_data.items():
    df_copy = future_df[["year", selected_indicator]].copy()
    df_copy["scenario"] = scenarios[scenario_name].description[:30]
    df_copy["scenario_id"] = scenario_name
    comparison_data.append(df_copy)

# Add historical data
historical = panel[panel["iso3"] == selected_country][["year", selected_indicator]].copy()
historical["scenario"] = "Historical"
historical["scenario_id"] = "historical"
comparison_data.insert(0, historical)

combined_df = pd.concat(comparison_data, ignore_index=True)

# Main visualization
st.subheader(f"{selected_indicator} Projections")

fig = px.line(
    combined_df,
    x="year",
    y=selected_indicator,
    color="scenario",
    title=f"{selected_indicator} - Scenario Comparison",
    markers=True,
    labels={"year": "Year", selected_indicator: selected_indicator, "scenario": "Scenario"}
)

# Add vertical line for current year
current_year = historical["year"].max()
fig.add_vline(x=current_year, line_dash="dash", line_color="gray", annotation_text="Current")

fig.update_layout(hovermode="x unified", height=500)
st.plotly_chart(fig, width='stretch')

# Scenario comparison table
st.subheader("Scenario Outcomes Comparison")

final_year = combined_df[combined_df["scenario_id"] != "historical"]["year"].max()
final_values = combined_df[
    (combined_df["year"] == final_year) & (combined_df["scenario_id"] != "historical")
]

# Calculate change from baseline
final_values = final_values.copy()  # Avoid SettingWithCopyWarning
if "baseline" in scenario_data:
    baseline_value = final_values[final_values["scenario_id"] == "baseline"][selected_indicator].iloc[0]
    final_values.loc[:, "vs_baseline"] = ((final_values[selected_indicator] - baseline_value) / baseline_value * 100).round(2)
else:
    final_values.loc[:, "vs_baseline"] = 0

# Calculate change from current
current_value = historical[selected_indicator].iloc[-1]
final_values.loc[:, "vs_current"] = ((final_values[selected_indicator] - current_value) / current_value * 100).round(2)

comparison_table = final_values[["scenario", selected_indicator, "vs_baseline", "vs_current"]]
comparison_table.columns = ["Scenario", f"Value ({int(final_year)})", "vs Baseline (%)", "vs Current (%)"]

st.dataframe(
    comparison_table.style.background_gradient(subset=["vs Baseline (%)", "vs Current (%)"], cmap="RdYlGn"),
    width='stretch'
)

# Multi-indicator comparison
st.subheader("Multi-Indicator Analysis")

selected_indicators_multi = st.multiselect(
    "Select multiple indicators for comparison",
    base_indicators,
    default=base_indicators[:3] if len(base_indicators) >= 3 else base_indicators
)

if selected_indicators_multi:
    cols = st.columns(min(3, len(selected_indicators_multi)))
    
    for idx, indicator in enumerate(selected_indicators_multi):
        col_idx = idx % 3
        
        with cols[col_idx]:
            # Generate data for this indicator
            multi_comparison_data = []
            for scenario_name, future_df in scenario_data.items():
                if indicator in future_df.columns:
                    final_value = future_df[indicator].iloc[-1]
                    multi_comparison_data.append({
                        "scenario": scenarios[scenario_name].description[:20],
                        "value": final_value
                    })
            
            if multi_comparison_data:
                multi_df = pd.DataFrame(multi_comparison_data)
                fig_bar = px.bar(
                    multi_df,
                    x="scenario",
                    y="value",
                    title=f"{indicator} ({int(final_year)})",
                    color="value",
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(showlegend=False, height=300, xaxis_title="")
                st.plotly_chart(fig_bar, width='stretch')

# Heatmap: all scenarios x multiple indicators
st.subheader("Scenario Impact Heatmap")

if len(selected_indicators_multi) >= 2:
    heatmap_data = []
    
    for scenario_name, future_df in scenario_data.items():
        row: dict[str, str | float] = {"Scenario": scenarios[scenario_name].description[:25]}
        
        for indicator in selected_indicators_multi:
            if indicator in future_df.columns:
                final_value = future_df[indicator].iloc[-1]
                current_val = historical[historical["year"] == historical["year"].max()][indicator].iloc[0] if indicator in historical.columns else final_value
                
                if current_val != 0:
                    pct_change = ((final_value - current_val) / abs(current_val)) * 100
                    row[indicator] = pct_change
                else:
                    row[indicator] = 0
        
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index("Scenario")
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale="RdYlGn",
        zmid=0,
        text=heatmap_df.values.round(1),
        texttemplate="%{text}%",
        textfont={"size": 10},
        colorbar=dict(title="% Change")
    ))
    
    fig_heatmap.update_layout(
        title="% Change from Current by Scenario",
        height=400,
        xaxis_title="Indicator",
        yaxis_title="Scenario"
    )
    
    st.plotly_chart(fig_heatmap, width='stretch')

# Scenario descriptions
st.subheader("Scenario Descriptions")

for scenario_name in selected_scenarios:
    with st.expander(f"{scenarios[scenario_name].name}"):
        st.write(scenarios[scenario_name].description)

# Export data
st.subheader("Export Projections")
csv = combined_df.to_csv(index=False)
st.download_button(
    label="Download Scenario Data",
    data=csv,
    file_name=f"scenarios_{selected_country}_{selected_horizon}y.csv",
    mime="text/csv"
)
