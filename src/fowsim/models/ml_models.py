from __future__ import annotations

import warnings
import os
from dataclasses import dataclass
from typing import Any, Optional
import logging

# Suppress sklearn and other warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*convergence.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    name: str
    model: Any
    description: str = ""


def make_models(random_state: int = 42, use_advanced: bool = True) -> list[ModelBundle]:
    """Create ensemble of ML models for time series forecasting.
    
    As per project proposal, includes:
    - Regression models (Ridge, Lasso, ElasticNet, Linear, SVR)
    - Tree-based models (RandomForest, GradientBoosting, DecisionTree)
    - Neural networks (MLP, TensorFlow/Keras)
    - Time-series models (ARIMA via baselines.py)
    - Clustering (for pattern detection - separate function)
    
    Args:
        random_state: Random seed for reproducibility
        use_advanced: Include TensorFlow and advanced models (default: True per proposal)
    
    Returns:
        List of ModelBundle objects with trained models
    """
    models = [
        # REGRESSION MODELS (as per proposal requirement)
        ModelBundle(
            name="linear_regression",
            model=LinearRegression(n_jobs=1),
            description="Ordinary Least Squares Linear Regression"
        ),
        ModelBundle(
            name="ridge",
            model=Ridge(alpha=1.0, random_state=random_state),
            description="Ridge regression with L2 regularization"
        ),
        ModelBundle(
            name="lasso",
            model=Lasso(alpha=0.1, random_state=random_state, max_iter=10000, tol=1e-3),
            description="Lasso regression with L1 regularization"
        ),
        ModelBundle(
            name="elastic_net",
            model=ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=10000, tol=1e-3),
            description="Elastic Net combining L1 and L2 regularization"
        ),
        ModelBundle(
            name="svr",
            model=SVR(kernel='rbf', C=1.0, epsilon=0.1),
            description="Support Vector Regression"
        ),
        
        # TREE-BASED MODELS
        ModelBundle(
            name="decision_tree",
            model=DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                random_state=random_state
            ),
            description="Decision Tree Regressor"
        ),
        ModelBundle(
            name="random_forest",
            model=RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=1,
            ),
            description="Random Forest ensemble method"
        ),
        ModelBundle(
            name="gradient_boosting",
            model=GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=random_state,
            ),
            description="Gradient Boosting for sequential error correction"
        ),
        
        # NEURAL NETWORK (Scikit-learn)
        ModelBundle(
            name="mlp",
            model=MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate="adaptive",
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
            ),
            description="Multi-layer Perceptron neural network"
        ),
    ]
    
    # Add TENSORFLOW models (as required by proposal)
    if use_advanced:
        try:
            from fowsim.models.tensorflow_models import create_tensorflow_model
            tf_model = create_tensorflow_model(random_state=random_state)
            models.append(tf_model)
            logger.info("TensorFlow neural network model added successfully")
        except Exception as e:
            logger.warning(f"TensorFlow model not available: {e}")
    
    # Add XGBoost if available
    if use_advanced:
        try:
            import xgboost as xgb
            models.append(
                ModelBundle(
                    name="xgboost",
                    model=xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        min_child_weight=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                    description="XGBoost extreme gradient boosting"
                )
            )
        except ImportError:
            logger.info("XGBoost not available. Install with: pip install xgboost")
    
    # Add LightGBM if available
    if use_advanced:
        try:
            import lightgbm as lgb
            models.append(
                ModelBundle(
                    name="lightgbm",
                    model=lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                    description="LightGBM gradient boosting framework"
                )
            )
        except ImportError:
            logger.info("LightGBM not available. Install with: pip install lightgbm")
    
    return models


def create_ensemble_model(models: list[ModelBundle], random_state: int = 42) -> ModelBundle:
    """Create a voting ensemble from multiple models.
    
    Args:
        models: List of model bundles to ensemble
        random_state: Random seed
    
    Returns:
        Ensemble ModelBundle
    """
    estimators = [(m.name, m.model) for m in models[:5]]  # Use top 5 models
    
    ensemble = VotingRegressor(
        estimators=estimators,
        n_jobs=-1
    )
    
    return ModelBundle(
        name="ensemble_voting",
        model=ensemble,
        description="Voting ensemble of multiple models"
    )


def create_stacked_model(base_models: list[ModelBundle], random_state: int = 42) -> ModelBundle:
    """Create a stacked ensemble with meta-learner.
    
    Args:
        base_models: Base models for stacking
        random_state: Random seed
    
    Returns:
        Stacked ModelBundle
    """
    try:
        from sklearn.ensemble import StackingRegressor
        
        estimators = [(m.name, m.model) for m in base_models[:5]]
        final_estimator = Ridge(alpha=1.0, random_state=random_state)
        
        stacked = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1
        )
        
        return ModelBundle(
            name="stacked_ensemble",
            model=stacked,
            description="Stacked ensemble with Ridge meta-learner"
        )
    except Exception as e:
        logger.warning(f"Stacked ensemble creation failed: {e}")
        return base_models[0]  # Fallback to first model
