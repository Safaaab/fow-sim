"""Unit tests for scenario simulations."""
import pytest
import pandas as pd

from fowsim.simulation.scenarios import (
    scenario_registry,
    RapidAIAdoption,
    RemoteWorkExpansion,
    BaselineScenario
)


def test_scenario_registry():
    """Test scenario registry returns all scenarios."""
    scenarios = scenario_registry()
    
    assert len(scenarios) >= 5, "Should have multiple scenarios"
    assert "baseline" in scenarios
    assert "rapid_ai" in scenarios
    assert all(hasattr(s, "apply") for s in scenarios.values())


def test_baseline_scenario():
    """Test baseline scenario doesn't modify data."""
    scenario = BaselineScenario()
    
    data = pd.DataFrame({
        "year": [2025, 2026, 2027],
        "internet_users": [70.0, 70.0, 70.0],
        "gdp_per_capita": [30000, 30000, 30000]
    })
    
    result = scenario.apply(data)
    
    # Baseline should not change values
    pd.testing.assert_frame_equal(result, data)


def test_rapid_ai_scenario():
    """Test rapid AI scenario increases technology indicators."""
    scenario = RapidAIAdoption()
    
    data = pd.DataFrame({
        "year": [2025, 2026, 2027, 2028, 2029],
        "internet_users": [70.0, 70.0, 70.0, 70.0, 70.0],
        "rd_spend_pct_gdp": [2.0, 2.0, 2.0, 2.0, 2.0]
    })
    
    result = scenario.apply(data)
    
    # Check that values increase over time
    assert result["internet_users"].iloc[-1] > result["internet_users"].iloc[0]
    assert result["rd_spend_pct_gdp"].iloc[-1] > result["rd_spend_pct_gdp"].iloc[0]


def test_remote_work_scenario():
    """Test remote work scenario modifies relevant indicators."""
    scenario = RemoteWorkExpansion()
    
    data = pd.DataFrame({
        "year": [2025, 2026, 2027],
        "internet_users": [70.0, 70.0, 70.0],
        "services_employment_share": [60.0, 60.0, 60.0]
    })
    
    result = scenario.apply(data)
    
    # Should increase digital infrastructure and services
    assert result["internet_users"].iloc[-1] > data["internet_users"].iloc[-1]
    assert result["services_employment_share"].iloc[-1] > data["services_employment_share"].iloc[-1]


def test_scenario_preserves_structure():
    """Test scenarios preserve DataFrame structure."""
    scenarios = scenario_registry()
    
    data = pd.DataFrame({
        "year": [2025, 2026, 2027],
        "indicator1": [10.0, 10.0, 10.0],
        "indicator2": [20.0, 20.0, 20.0]
    })
    
    for scenario in scenarios.values():
        result = scenario.apply(data)
        
        assert list(result.columns) == list(data.columns), "Columns should be preserved"
        assert len(result) == len(data), "Row count should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
