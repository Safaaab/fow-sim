from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (as required by proposal)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error (as required by proposal)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """R-squared (Coefficient of Determination) - Accuracy metric.
    
    As required by proposal: "Validating results using metrics such as RMSE and accuracy"
    R² represents the proportion of variance explained by the model (accuracy).
    
    Returns:
        R² score between -inf and 1.0 (1.0 = perfect prediction)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - (ss_res / ss_tot))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error.
    
    Useful for understanding error as a percentage of actual values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return 0.0
    
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def accuracy_percentage(y_true, y_pred, tolerance=0.1) -> float:
    """Calculate accuracy as percentage of predictions within tolerance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        tolerance: Acceptable error as fraction of true value (default 10%)
        
    Returns:
        Percentage of predictions within tolerance (0-100)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    errors = np.abs(y_true - y_pred)
    acceptable = errors <= (np.abs(y_true) * tolerance)
    
    return float(np.mean(acceptable) * 100)

