"""Unit tests for model training and evaluation."""
import pytest
import pandas as pd
import numpy as np

from fowsim.models.ml_models import make_models, create_ensemble_model
from fowsim.models.metrics import rmse, mae
from fowsim.models.baselines import naive_forecast, drift_forecast


def test_make_models():
    """Test model creation."""
    models = make_models(random_state=42, use_advanced=False)
    
    # Should have at least 9 models (proposal requires multiple regression, neural nets, etc.)
    assert len(models) >= 9, f"Should create at least 9 models, got {len(models)}"
    assert all(hasattr(m.model, "fit") for m in models), "All models should have fit method"
    assert all(hasattr(m.model, "predict") for m in models), "All models should have predict method"


def test_rmse():
    """Test RMSE calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    result = rmse(y_true, y_pred)
    
    assert isinstance(result, float)
    assert result > 0
    assert result < 1  # Should be small for close predictions


def test_mae():
    """Test MAE calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    result = mae(y_true, y_pred)
    
    assert isinstance(result, float)
    assert result > 0
    assert result < 1


def test_naive_forecast():
    """Test naive baseline forecast."""
    series = pd.Series([10, 12, 11, 13, 15])
    forecast = naive_forecast(series, steps=3)
    
    assert len(forecast) == 3
    assert all(forecast == 15), "Naive forecast should repeat last value"


def test_drift_forecast():
    """Test drift baseline forecast."""
    series = pd.Series([10, 12, 14, 16, 18])  # Linear trend
    forecast = drift_forecast(series, steps=3)
    
    assert len(forecast) == 3
    assert forecast[0] > series.iloc[-1], "Drift should continue trend"


def test_model_fit_predict():
    """Test model can fit and predict."""
    models = make_models(random_state=42, use_advanced=False)
    
    # Simple synthetic data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    for model_bundle in models[:2]:  # Test first 2 models
        model = model_bundle.model
        model.fit(X, y)
        pred = model.predict(X)
        
        assert len(pred) == len(y)
        assert isinstance(pred, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
