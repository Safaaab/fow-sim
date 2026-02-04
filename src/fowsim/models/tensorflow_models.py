"""
TensorFlow/Keras Neural Network Models for Future of Work Prediction

As required by the project proposal:
- "Building predictive models using regression, clustering, and time-series forecasting 
  (Scikit-learn, TensorFlow)"
- "Libraries: Pandas, Scikit-learn, TensorFlow, Matplotlib, Plotly"

This module implements deep learning models using TensorFlow/Keras for time-series forecasting.
"""

from __future__ import annotations

import logging
import warnings
import os
import numpy as np
from dataclasses import dataclass
from typing import Any

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

logger = logging.getLogger(__name__)


class KerasRegressorWrapper:
    """Wrapper to make Keras models compatible with scikit-learn API."""
    
    def __init__(self, model, epochs=50, batch_size=32, verbose=0):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.history_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None
        self._input_dim = None
        
    def fit(self, X, y, **kwargs):
        """Train the Keras model."""
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        
        # Convert to numpy and handle NaN values
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        
        # Replace NaN/inf with 0 for stability
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features for better neural network training
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()
        
        X_scaled = self.scaler_X_.fit_transform(X_arr)
        y_scaled = self.scaler_y_.fit_transform(y_arr).flatten()
        
        # Handle any remaining NaN after scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Rebuild model with correct input dimension if needed
        self._input_dim = X_scaled.shape[1]
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history_ = self.model.fit(
            X_scaled, y_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[early_stop],
            validation_split=0.1
        )
        return self
        
    def predict(self, X):
        """Make predictions."""
        # Convert to numpy and handle NaN values
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply same scaling used during training
        if self.scaler_X_ is not None:
            X_arr = self.scaler_X_.transform(X_arr)
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get predictions
        y_pred_scaled = self.model.predict(X_arr, verbose=0).flatten()
        
        # Inverse transform predictions to original scale
        if self.scaler_y_ is not None:
            y_pred = self.scaler_y_.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            y_pred = y_pred_scaled
            
        return np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def build_dense_nn(input_dim: int, random_state: int = 42) -> Any:
    """Build a dense feedforward neural network using TensorFlow/Keras.
    
    Architecture:
    - Input layer
    - Dense(128) + ReLU + Dropout(0.3)
    - Dense(64) + ReLU + Dropout(0.2)
    - Dense(32) + ReLU
    - Output layer (1 neuron for regression)
    
    Args:
        input_dim: Number of input features
        random_state: Random seed for reproducibility
        
    Returns:
        Compiled Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Set seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu', 
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)  # Output layer for regression
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
        
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        raise


def build_lstm_model(input_dim: int, random_state: int = 42) -> Any:
    """Build an LSTM model for time-series forecasting.
    
    Note: This is a simplified LSTM that works with tabular data by treating
    each feature as a timestep (for compatibility with current pipeline).
    
    Args:
        input_dim: Number of input features
        random_state: Random seed
        
    Returns:
        Compiled Keras LSTM model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim, 1)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
        
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        raise


class LSTMWrapper(KerasRegressorWrapper):
    """Special wrapper for LSTM that reshapes input data."""
    
    def fit(self, X, y, **kwargs):
        """Reshape X for LSTM and train."""
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        return super().fit(X_reshaped, y, **kwargs)
        
    def predict(self, X):
        """Reshape X for LSTM and predict."""
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X_reshaped, verbose=0).flatten()


def create_tensorflow_model(random_state: int = 42, model_type: str = "dense"):
    """Create a TensorFlow model compatible with the training pipeline.
    
    This function is called by ml_models.py to add TensorFlow models to the ensemble.
    The model uses lazy initialization to avoid building until we know input dimensions.
    
    Args:
        random_state: Random seed for reproducibility
        model_type: Type of model - "dense" or "lstm"
        
    Returns:
        ModelBundle with TensorFlow model
    """
    from fowsim.models.ml_models import ModelBundle
    
    class LazyTensorFlowModel:
        """Lazy initialization wrapper for TensorFlow models."""
        
        def __init__(self, random_state, model_type):
            self.random_state = random_state
            self.model_type = model_type
            self.model = None
            self._last_input_dim = None
            
        def fit(self, X, y, **kwargs):
            """Build model only when input dimensions change."""
            import numpy as np
            import tensorflow as tf
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            X_arr = np.asarray(X, dtype=np.float32)
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            input_dim = X_arr.shape[1]
            
            # Only rebuild model if input dimensions changed
            if self.model is None or self._last_input_dim != input_dim:
                tf.keras.backend.clear_session()  # Clear previous models
                if self.model_type == "lstm":
                    keras_model = build_lstm_model(input_dim, self.random_state)
                    self.model = LSTMWrapper(keras_model, epochs=30, batch_size=32, verbose=0)
                else:
                    keras_model = build_dense_nn(input_dim, self.random_state)
                    self.model = KerasRegressorWrapper(keras_model, epochs=30, batch_size=32, verbose=0)
                self._last_input_dim = input_dim
            
            return self.model.fit(X_arr, y, **kwargs)
            
        def predict(self, X):
            """Make predictions."""
            if self.model is None:
                raise ValueError("Model must be fitted before making predictions")
            return self.model.predict(X)
            
        def get_params(self, deep=True):
            return {'random_state': self.random_state, 'model_type': self.model_type}
            
        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self
    
    lazy_model = LazyTensorFlowModel(random_state, model_type)
    
    return ModelBundle(
        name="tensorflow_nn",
        model=lazy_model,
        description="TensorFlow/Keras Deep Neural Network (as per proposal requirement)"
    )


def create_clustering_analyzer(n_clusters: int = 5, random_state: int = 42):
    """Create clustering models for pattern detection in workforce data.
    
    As per proposal: "Building predictive models using regression, clustering, 
    and time-series forecasting"
    
    This is used separately from regression models for:
    - Identifying country clusters with similar work patterns
    - Detecting anomalies in workforce trends
    - Segmenting markets for targeted predictions
    
    Args:
        n_clusters: Number of clusters for K-Means
        random_state: Random seed
        
    Returns:
        Dictionary of clustering models
    """
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    
    return {
        'kmeans': KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
        'dbscan': DBSCAN(eps=0.5, min_samples=5),
        'scaler': StandardScaler()
    }
