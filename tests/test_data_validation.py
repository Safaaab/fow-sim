"""Unit tests for data validation module."""
import pytest
import pandas as pd
import numpy as np

from fowsim.data.validate import validate_panel


def test_validate_panel_valid():
    """Test validation with valid panel data."""
    data = {
        "iso3": ["USA", "USA", "GBR", "GBR"],
        "year": [2020, 2021, 2020, 2021],
        "gdp": [100, 105, 80, 82],
        "unemployment": [5.0, 4.5, 6.0, 5.8]
    }
    panel = pd.DataFrame(data)
    
    # Should not raise any exception
    validate_panel(panel)


def test_validate_panel_missing_columns():
    """Test validation fails with missing required columns."""
    data = {
        "country": ["USA", "GBR"],
        "gdp": [100, 80]
    }
    panel = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="missing required columns"):
        validate_panel(panel)


def test_validate_panel_nan_years():
    """Test validation fails with NaN years."""
    data = {
        "iso3": ["USA", "GBR"],
        "year": [2020, np.nan],
        "gdp": [100, 80]
    }
    panel = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="NaN years"):
        validate_panel(panel)


def test_validate_panel_non_monotonic():
    """Test validation fails with non-increasing years."""
    data = {
        "iso3": ["USA", "USA", "USA"],
        "year": [2020, 2021, 2020],  # Not monotonic
        "gdp": [100, 105, 98]
    }
    panel = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="Non-increasing years"):
        validate_panel(panel)


def test_validate_panel_no_numeric():
    """Test validation fails with no numeric columns."""
    data = {
        "iso3": ["USA", "GBR"],
        "year": ["2020", "2020"],  # Make year non-numeric
        "name": ["United States", "United Kingdom"]
    }
    panel = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="No numeric columns"):
        validate_panel(panel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
