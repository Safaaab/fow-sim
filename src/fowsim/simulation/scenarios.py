from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        """Modify future feature trajectories (in-place copy)."""
        return future_features


class BaselineScenario(Scenario):
    """No changes - business as usual trajectory."""
    
    def __init__(self) -> None:
        super().__init__(
            name="baseline",
            description="Business as usual - no significant changes from current trends"
        )


class RapidAIAdoption(Scenario):
    """Accelerated AI and automation adoption across industries."""
    
    def __init__(self) -> None:
        super().__init__(
            name="rapid_ai",
            description="Rapid AI adoption: 15% annual increase in digital indicators, "
                       "boosted R&D spending, and accelerated innovation"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years_from_start = df["year"] - df["year"].min()
        
        # Aggressive technology adoption
        if "internet_users" in df.columns:
            df["internet_users"] = df["internet_users"] * (1 + 0.15 * years_from_start)
            df["internet_users"] = df["internet_users"].clip(upper=100)  # Cap at 100%
        
        if "rd_spend_pct_gdp" in df.columns:
            df["rd_spend_pct_gdp"] = df["rd_spend_pct_gdp"] + 0.05 * years_from_start
        
        if "tertiary_enrollment" in df.columns:
            df["tertiary_enrollment"] = df["tertiary_enrollment"] + 0.3 * years_from_start
        
        if "high_tech_exports" in df.columns:
            df["high_tech_exports"] = df["high_tech_exports"] * (1 + 0.12 * years_from_start)
        
        # Increased automation risk
        if "automation_risk" in df.columns:
            df["automation_risk"] = df["automation_risk"] + 0.02 * years_from_start
        
        return df


class RemoteWorkExpansion(Scenario):
    """Massive shift to remote and hybrid work models."""
    
    def __init__(self) -> None:
        super().__init__(
            name="remote_work_expansion",
            description="70% of knowledge workers shift to remote/hybrid models, "
                       "boosting digital infrastructure and services employment"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years_from_start = df["year"] - df["year"].min()
        
        # Digital infrastructure boost
        if "internet_users" in df.columns:
            df["internet_users"] = df["internet_users"] + 0.8 * years_from_start
            df["internet_users"] = df["internet_users"].clip(upper=100)
        
        # Services sector growth
        if "services_employment_share" in df.columns:
            df["services_employment_share"] = df["services_employment_share"] + 0.15 * years_from_start
            df["services_employment_share"] = df["services_employment_share"].clip(upper=90)
        
        # Urban decentralization (negative growth)
        if "urban_population" in df.columns:
            df["urban_population"] = df["urban_population"] - 0.05 * years_from_start
        
        # Remote work adoption
        if "remote_work_percentage" in df.columns:
            df["remote_work_percentage"] = df["remote_work_percentage"] + 3.5 * years_from_start
            df["remote_work_percentage"] = df["remote_work_percentage"].clip(upper=70)
        
        return df


class EconomicRecession(Scenario):
    """Global economic downturn with lasting impacts."""
    
    def __init__(self) -> None:
        super().__init__(
            name="recession_shock",
            description="Severe 2-year recession: GDP down 8%, unemployment up 4%, "
                       "gradual recovery over 5 years"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years = sorted(df["year"].unique())
        
        if len(years) < 2:
            return df
        
        # Severe shock in first 2 years
        shock_years = set(years[:2])
        recovery_years = set(years[2:7]) if len(years) > 2 else set()
        
        for idx, row in df.iterrows():
            year = row["year"]
            
            if year in shock_years:
                # Severe impact
                if "gdp_per_capita" in df.columns:
                    df.at[idx, "gdp_per_capita"] *= 0.92  # -8%
                if "unemployment_rate" in df.columns:
                    df.at[idx, "unemployment_rate"] += 4.0
                if "labor_force_participation" in df.columns:
                    df.at[idx, "labor_force_participation"] -= 2.5
                if "gdp_growth" in df.columns:
                    df.at[idx, "gdp_growth"] = -5.0
                    
            elif year in recovery_years:
                # Gradual recovery
                years_since_shock = years.index(year) - 1
                recovery_factor = min(years_since_shock / 5, 1.0)
                
                if "gdp_per_capita" in df.columns:
                    df.at[idx, "gdp_per_capita"] *= (1 + 0.015 * recovery_factor)
                if "unemployment_rate" in df.columns:
                    df.at[idx, "unemployment_rate"] += 2.0 * (1 - recovery_factor)
        
        return df


class SkillsGapCrisis(Scenario):
    """Widening skills gap with inadequate education response."""
    
    def __init__(self) -> None:
        super().__init__(
            name="skills_gap_crisis",
            description="Mismatch between education and market needs, "
                       "leading to high unemployment despite job openings"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years_from_start = df["year"] - df["year"].min()
        
        # Education lags behind
        if "tertiary_enrollment" in df.columns:
            df["tertiary_enrollment"] = df["tertiary_enrollment"] + 0.05 * years_from_start
        
        # But unemployment rises due to mismatch
        if "unemployment_rate" in df.columns:
            df["unemployment_rate"] = df["unemployment_rate"] + 0.3 * years_from_start
        
        # Technology still advances
        if "rd_spend_pct_gdp" in df.columns:
            df["rd_spend_pct_gdp"] = df["rd_spend_pct_gdp"] + 0.02 * years_from_start
        
        # Vulnerable employment increases
        if "vulnerable_employment" in df.columns:
            df["vulnerable_employment"] = df["vulnerable_employment"] + 0.4 * years_from_start
        
        return df


class GreenTransition(Scenario):
    """Rapid transition to green economy and sustainable work practices."""
    
    def __init__(self) -> None:
        super().__init__(
            name="green_transition",
            description="Massive investment in green tech, remote work for carbon reduction, "
                       "retraining programs for fossil fuel workers"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years_from_start = df["year"] - df["year"].min()
        
        # R&D investment in green tech
        if "rd_spend_pct_gdp" in df.columns:
            df["rd_spend_pct_gdp"] = df["rd_spend_pct_gdp"] + 0.04 * years_from_start
        
        # Shift from industry to services
        if "industry_employment_share" in df.columns:
            df["industry_employment_share"] = df["industry_employment_share"] - 0.2 * years_from_start
        
        if "services_employment_share" in df.columns:
            df["services_employment_share"] = df["services_employment_share"] + 0.2 * years_from_start
        
        # Education investment
        if "education_expenditure" in df.columns:
            df["education_expenditure"] = df["education_expenditure"] + 0.1 * years_from_start
        
        # Remote work increases
        if "remote_work_percentage" in df.columns:
            df["remote_work_percentage"] = df["remote_work_percentage"] + 2.5 * years_from_start
        
        return df


class HybridWorkDominance(Scenario):
    """Hybrid work becomes the global standard."""
    
    def __init__(self) -> None:
        super().__init__(
            name="hybrid_work_standard",
            description="3-2 hybrid model becomes global norm, "
                       "reshaping urban planning and real estate"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years_from_start = df["year"] - df["year"].min()
        
        # Moderate remote work adoption (not as extreme as full remote)
        if "remote_work_percentage" in df.columns:
            # Converge to ~60% hybrid
            df["remote_work_percentage"] = df["remote_work_percentage"] + 2.0 * years_from_start
            df["remote_work_percentage"] = df["remote_work_percentage"].clip(upper=60)
        
        # Balanced infrastructure needs
        if "internet_users" in df.columns:
            df["internet_users"] = df["internet_users"] + 0.5 * years_from_start
        
        # Stable employment patterns
        if "labor_force_participation" in df.columns:
            df["labor_force_participation"] = df["labor_force_participation"] + 0.15 * years_from_start
        
        # Services grow moderately
        if "services_employment_share" in df.columns:
            df["services_employment_share"] = df["services_employment_share"] + 0.1 * years_from_start
        
        return df


class EducationRevolution(Scenario):
    """Massive global investment in education and reskilling."""
    
    def __init__(self) -> None:
        super().__init__(
            name="education_revolution",
            description="Universal access to tertiary education and continuous reskilling programs, "
                       "preparing workforce for technological changes"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years_from_start = df["year"] - df["year"].min()
        
        # Massive enrollment increase
        if "tertiary_enrollment" in df.columns:
            df["tertiary_enrollment"] = df["tertiary_enrollment"] + 0.5 * years_from_start
            df["tertiary_enrollment"] = df["tertiary_enrollment"].clip(upper=95)
        
        # Education spending surge
        if "education_expenditure" in df.columns:
            df["education_expenditure"] = df["education_expenditure"] + 0.15 * years_from_start
        
        # Literacy improvements
        if "adult_literacy" in df.columns:
            df["adult_literacy"] = df["adult_literacy"] + 0.2 * years_from_start
            df["adult_literacy"] = df["adult_literacy"].clip(upper=100)
        
        # Lower unemployment due to better skills match
        if "unemployment_rate" in df.columns:
            df["unemployment_rate"] = df["unemployment_rate"] - 0.2 * years_from_start
            df["unemployment_rate"] = df["unemployment_rate"].clip(lower=2)
        
        # Higher productivity
        if "gdp_per_capita" in df.columns:
            df["gdp_per_capita"] = df["gdp_per_capita"] * (1 + 0.02 * years_from_start)
        
        return df


class AutomationSurge(Scenario):
    """Extreme automation replacing 40% of current jobs by 2045."""
    
    def __init__(self) -> None:
        super().__init__(
            name="automation_surge",
            description="Massive automation wave: robots and AI replace routine jobs, "
                       "creating structural unemployment without retraining"
        )

    def apply(self, future_features: pd.DataFrame) -> pd.DataFrame:
        df = future_features.copy()
        years_from_start = df["year"] - df["year"].min()
        
        # High automation adoption
        if "automation_risk" in df.columns:
            df["automation_risk"] = df["automation_risk"] + 0.03 * years_from_start
        
        # AI adoption surge
        if "ai_adoption_index" in df.columns:
            df["ai_adoption_index"] = df["ai_adoption_index"] * (1 + 0.2 * years_from_start)
        
        # Unemployment rises (without adequate transition support)
        if "unemployment_rate" in df.columns:
            df["unemployment_rate"] = df["unemployment_rate"] + 0.5 * years_from_start
        
        # Industry jobs decline sharply
        if "industry_employment_share" in df.columns:
            df["industry_employment_share"] = df["industry_employment_share"] - 0.4 * years_from_start
        
        # Services partially absorb
        if "services_employment_share" in df.columns:
            df["services_employment_share"] = df["services_employment_share"] + 0.2 * years_from_start
        
        # Technology exports boom
        if "high_tech_exports" in df.columns:
            df["high_tech_exports"] = df["high_tech_exports"] * (1 + 0.15 * years_from_start)
        
        return df


def scenario_registry() -> dict[str, Scenario]:
    """Registry of all available scenarios.
    
    Returns:
        Dictionary mapping scenario names to Scenario objects
    """
    scenarios = [
        BaselineScenario(),
        RapidAIAdoption(),
        RemoteWorkExpansion(),
        EconomicRecession(),
        SkillsGapCrisis(),
        GreenTransition(),
        HybridWorkDominance(),
        EducationRevolution(),
        AutomationSurge(),
    ]
    return {s.name: s for s in scenarios}


def get_scenario_categories() -> dict[str, list[str]]:
    """Group scenarios by category for UI organization."""
    return {
        "Technology-Driven": ["rapid_ai", "automation_surge"],
        "Work Model Shifts": ["remote_work_expansion", "hybrid_work_standard"],
        "Economic": ["recession_shock", "green_transition"],
        "Social & Education": ["skills_gap_crisis", "education_revolution"],
        "Baseline": ["baseline"],
    }
