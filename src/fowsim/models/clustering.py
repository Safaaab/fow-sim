"""
Clustering Analysis for Future of Work Data

As required by the project proposal:
"Building predictive models using regression, clustering, and time-series forecasting"

This module implements clustering algorithms to:
1. Identify country groups with similar workforce patterns
2. Detect anomalies in workforce trends
3. Segment markets for targeted predictions
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Container for clustering analysis results."""
    algorithm: str
    labels: np.ndarray
    n_clusters: int
    cluster_centers: np.ndarray | None = None
    silhouette_score: float | None = None
    features_used: list[str] | None = None


def perform_clustering_analysis(
    panel_df: pd.DataFrame,
    algorithm: str = "kmeans",
    n_clusters: int = 5,
    features: list[str] | None = None
) -> ClusteringResult:
    """Perform clustering analysis on workforce data.
    
    This addresses the proposal requirement for "clustering" as one of the main
    predictive modeling approaches alongside regression and time-series.
    
    Args:
        panel_df: Panel dataset with country-year observations
        algorithm: Clustering algorithm - "kmeans", "dbscan", or "hierarchical"
        n_clusters: Number of clusters (for kmeans and hierarchical)
        features: Features to use for clustering. If None, uses all numeric columns.
        
    Returns:
        ClusteringResult with cluster labels and metadata
    """
    logger.info(f"Starting {algorithm} clustering analysis with {n_clusters} clusters")
    
    # Select features
    if features is None:
        numeric_cols = panel_df.select_dtypes(include=[np.number]).columns
        # Exclude iso3 year and target variables for feature selection
        exclude = ['year', 'iso3', 'unemployment_rate', 'labor_force_participation',
                  'gdp_per_capita', 'services_employment_share', 'industry_employment_share',
                  'agriculture_employment_share', 'wage_employees', 'vulnerable_employment']
        features = [c for c in numeric_cols if c not in exclude and not c.endswith('_lag1') 
                   and not c.endswith('_lag2') and not c.endswith('_lag3')]
        features = features[:20]  # Limit to top 20 features for interpretability
    
    logger.info(f"Using {len(features)} features for clustering")
    
    # Prepare data
    X = panel_df[features].fillna(0)
    
    # Standardize features (crucial for distance-based clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering based on algorithm
    if algorithm == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)
        centers = scaler.inverse_transform(clusterer.cluster_centers_)
        
    elif algorithm == "dbscan":
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        centers = None
        
    elif algorithm == "hierarchical":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X_scaled)
        centers = None
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Calculate silhouette score (quality metric)
    silhouette = None
    if len(set(labels)) > 1:
        try:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X_scaled, labels)
            logger.info(f"Silhouette score: {silhouette:.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
    
    logger.info(f"Clustering complete: {len(set(labels))} clusters found")
    
    return ClusteringResult(
        algorithm=algorithm,
        labels=labels,
        n_clusters=len(set(labels)),
        cluster_centers=centers,
        silhouette_score=silhouette,
        features_used=features
    )


def cluster_countries_by_patterns(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster countries by their workforce and technology patterns.
    
    This creates country segments that can be used for:
    - Targeted predictions (train separate models per cluster)
    - Comparative analysis (see which countries follow similar patterns)
    - Anomaly detection (identify countries deviating from their cluster)
    
    Args:
        panel_df: Panel dataset
        
    Returns:
        DataFrame with countries and their cluster assignments
    """
    # Aggregate to country level (average across years)
    country_profiles = panel_df.groupby('iso3').mean(numeric_only=True)
    
    # Select technology and workforce indicators
    tech_indicators = [
        'internet_users', 'mobile_subscriptions', 'rd_spend_pct_gdp',
        'patent_applications', 'scientific_articles', 'high_tech_exports'
    ]
    workforce_indicators = [
        'labor_force_participation', 'unemployment_rate',
        'services_employment_share', 'industry_employment_share',
        'tertiary_enrollment', 'female_labor_participation'
    ]
    
    # Available features (handle missing columns)
    available_features = [f for f in tech_indicators + workforce_indicators 
                         if f in country_profiles.columns]
    
    if len(available_features) < 3:
        logger.warning("Not enough features for clustering")
        return pd.DataFrame({'iso3': panel_df['iso3'].unique(), 'cluster': 0})
    
    X = country_profiles[available_features].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    optimal_k = min(5, len(country_profiles) // 3)  # Don't over-cluster
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'iso3': country_profiles.index,
        'cluster': clusters
    })
    
    logger.info(f"Countries clustered into {optimal_k} groups")
    
    return result


def apply_pca_for_visualization(panel_df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Apply PCA for dimensionality reduction and visualization.
    
    Useful for:
    - Visualizing high-dimensional data in 2D/3D
    - Feature reduction before clustering
    - Understanding principal patterns in the data
    
    Args:
        panel_df: Panel dataset
        n_components: Number of principal components (2 or 3 for visualization)
        
    Returns:
        DataFrame with original data plus PCA components
    """
    numeric_cols = panel_df.select_dtypes(include=[np.number]).columns
    exclude = ['year']
    features = [c for c in numeric_cols if c not in exclude][:30]
    
    X = panel_df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)
    
    # Add to DataFrame
    result = panel_df.copy()
    for i in range(n_components):
        result[f'PC{i+1}'] = components[:, i]
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return result


def save_clustering_results(clustering_result: ClusteringResult, 
                            panel_df: pd.DataFrame,
                            output_path: Path) -> None:
    """Save clustering results to file.
    
    Args:
        clustering_result: Clustering results
        panel_df: Original panel data
        output_path: Path to save results
    """
    # Add cluster labels to panel
    result_df = panel_df.copy()
    result_df['cluster'] = clustering_result.labels
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    logger.info(f"Clustering results saved to {output_path}")
    
    # Print cluster summary
    print(f"\n{'='*80}")
    print(f"CLUSTERING ANALYSIS SUMMARY ({clustering_result.algorithm.upper()})")
    print(f"{'='*80}")
    print(f"Number of clusters: {clustering_result.n_clusters}")
    if clustering_result.silhouette_score:
        print(f"Silhouette score: {clustering_result.silhouette_score:.3f}")
    
    # Cluster sizes
    cluster_sizes = pd.Series(clustering_result.labels).value_counts().sort_index()
    print(f"\nCluster sizes:")
    for cluster_id, size in cluster_sizes.items():
        print(f"  Cluster {cluster_id}: {size} observations")
    
    print(f"{'='*80}\n")
