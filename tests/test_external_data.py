"""Tests for external dataset loading and integration.

This module tests the external datasets required by the project proposal:
- automation_risk.csv: Automation probability by industry/occupation
- remote_work_trends.csv: Remote/hybrid work adoption metrics
- ai_adoption_index.csv: Enterprise AI/ML adoption scores
- skills_gap_data.csv: Skills gap and retraining metrics
"""

import pytest
import pandas as pd
from pathlib import Path

# Path to test data
DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# Expected countries from indicators.yaml (25 countries)
EXPECTED_COUNTRIES = [
    "USA", "GBR", "DEU", "JPN", "CAN", "AUS", "FRA", "ITA",  # Developed
    "CHN", "IND", "BRA", "RUS", "MEX", "IDN", "TUR", "KOR",  # Emerging
    "PAK", "BGD", "LKA", "NPL",  # South Asia
    "SAU", "ARE",  # Middle East
    "ZAF", "NGA", "EGY"  # Africa
]

EXPECTED_YEARS = list(range(2015, 2025))


class TestAutomationRiskData:
    """Test automation_risk.csv dataset."""
    
    @pytest.fixture
    def df(self):
        filepath = DATA_RAW / "automation_risk.csv"
        if not filepath.exists():
            pytest.skip("automation_risk.csv not found")
        return pd.read_csv(filepath)
    
    def test_required_columns_exist(self, df):
        """Check required columns are present."""
        required = ["iso3", "year", "automation_risk"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_country_coverage(self, df):
        """Verify all expected countries are present."""
        missing = set(EXPECTED_COUNTRIES) - set(df["iso3"].unique())
        assert len(missing) == 0, f"Missing countries: {missing}"
    
    def test_year_range(self, df):
        """Verify year range coverage."""
        years = df["year"].unique()
        assert min(years) <= 2015, "Data should start from 2015 or earlier"
        assert max(years) >= 2023, "Data should extend to at least 2023"
    
    def test_automation_risk_values(self, df):
        """Automation risk should be between 0 and 1."""
        assert df["automation_risk"].min() >= 0, "Automation risk cannot be negative"
        assert df["automation_risk"].max() <= 1, "Automation risk cannot exceed 1"
    
    def test_no_duplicate_records(self, df):
        """Each country-year combination should be unique."""
        duplicates = df.duplicated(subset=["iso3", "year"]).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate records"


class TestRemoteWorkTrendsData:
    """Test remote_work_trends.csv dataset."""
    
    @pytest.fixture
    def df(self):
        filepath = DATA_RAW / "remote_work_trends.csv"
        if not filepath.exists():
            pytest.skip("remote_work_trends.csv not found")
        return pd.read_csv(filepath)
    
    def test_required_columns_exist(self, df):
        """Check required columns are present."""
        required = ["iso3", "year", "remote_work_percentage"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_country_coverage(self, df):
        """Verify all expected countries are present."""
        missing = set(EXPECTED_COUNTRIES) - set(df["iso3"].unique())
        assert len(missing) == 0, f"Missing countries: {missing}"
    
    def test_percentage_values(self, df):
        """Percentages should be between 0 and 100."""
        pct_cols = [c for c in df.columns if "percentage" in c.lower() or "pct" in c.lower()]
        for col in pct_cols:
            assert df[col].min() >= 0, f"{col} cannot be negative"
            assert df[col].max() <= 100, f"{col} cannot exceed 100"
    
    def test_no_duplicate_records(self, df):
        """Each country-year combination should be unique."""
        duplicates = df.duplicated(subset=["iso3", "year"]).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate records"


class TestAiAdoptionIndexData:
    """Test ai_adoption_index.csv dataset."""
    
    @pytest.fixture
    def df(self):
        filepath = DATA_RAW / "ai_adoption_index.csv"
        if not filepath.exists():
            pytest.skip("ai_adoption_index.csv not found")
        return pd.read_csv(filepath)
    
    def test_required_columns_exist(self, df):
        """Check required columns are present."""
        required = ["iso3", "year", "ai_adoption_index"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_country_coverage(self, df):
        """Verify all expected countries are present."""
        missing = set(EXPECTED_COUNTRIES) - set(df["iso3"].unique())
        assert len(missing) == 0, f"Missing countries: {missing}"
    
    def test_index_values(self, df):
        """AI adoption index should be between 0 and 100."""
        assert df["ai_adoption_index"].min() >= 0, "Index cannot be negative"
        assert df["ai_adoption_index"].max() <= 100, "Index cannot exceed 100"
    
    def test_no_duplicate_records(self, df):
        """Each country-year combination should be unique."""
        duplicates = df.duplicated(subset=["iso3", "year"]).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate records"


class TestSkillsGapData:
    """Test skills_gap_data.csv dataset."""
    
    @pytest.fixture
    def df(self):
        filepath = DATA_RAW / "skills_gap_data.csv"
        if not filepath.exists():
            pytest.skip("skills_gap_data.csv not found")
        return pd.read_csv(filepath)
    
    def test_required_columns_exist(self, df):
        """Check required columns are present."""
        required = ["iso3", "year", "skills_gap_index"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_country_coverage(self, df):
        """Verify all expected countries are present."""
        missing = set(EXPECTED_COUNTRIES) - set(df["iso3"].unique())
        assert len(missing) == 0, f"Missing countries: {missing}"
    
    def test_index_values(self, df):
        """Skills gap index should be between 0 and 100."""
        assert df["skills_gap_index"].min() >= 0, "Index cannot be negative"
        assert df["skills_gap_index"].max() <= 100, "Index cannot exceed 100"
    
    def test_no_duplicate_records(self, df):
        """Each country-year combination should be unique."""
        duplicates = df.duplicated(subset=["iso3", "year"]).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate records"


class TestExternalDataIntegration:
    """Test external data integration into panel."""
    
    def test_all_datasets_exist(self):
        """Verify all required external datasets exist."""
        required_files = [
            "automation_risk.csv",
            "remote_work_trends.csv",
            "ai_adoption_index.csv",
            "skills_gap_data.csv"
        ]
        for filename in required_files:
            filepath = DATA_RAW / filename
            assert filepath.exists(), f"Missing external dataset: {filename}"
    
    def test_datasets_can_be_merged(self):
        """Test that all datasets can be merged on iso3 and year."""
        dfs = []
        for filename in ["automation_risk.csv", "remote_work_trends.csv", 
                         "ai_adoption_index.csv", "skills_gap_data.csv"]:
            filepath = DATA_RAW / filename
            if filepath.exists():
                dfs.append(pd.read_csv(filepath))
        
        if len(dfs) < 4:
            pytest.skip("Not all external datasets available")
        
        # Merge all on iso3 and year
        merged = dfs[0]
        for df in dfs[1:]:
            merge_cols = [c for c in df.columns if c not in ["country"]]
            merged = merged.merge(df[merge_cols], on=["iso3", "year"], how="outer")
        
        # Should have data for all countries
        assert len(merged["iso3"].unique()) >= len(EXPECTED_COUNTRIES) - 2
        
        # Should have consistent year range
        assert merged["year"].min() <= 2015
        assert merged["year"].max() >= 2023


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
