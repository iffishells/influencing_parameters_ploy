import numpy as np
from sklearn.metrics import r2_score

def calculate_errors(actual, predicted):
    """
    Calculate MAE, MSE, and RMSE.

    Parameters:
    - actual: array-like, actual values
    - predicted: array-like, predicted values

    Returns:
    - mae: float, Mean Absolute Error
    - mse: float, Mean Squared Error
    - rmse: float, Root Mean Squared Error
    """
    mae = np.round(np.mean(np.abs(actual - predicted)),2)
    mse = np.round(np.mean((actual - predicted)**2),2)
    rmse = np.round(np.sqrt(mse),2)
    mape = np.round(100 * (abs(actual - predicted) / actual).mean())
    r2 = np.round(r2_score(actual, predicted),2)
    error_metrics = {
            'mae':mae,
            'mse':mse,
            'rmse':rmse,
            'mape':mape,
            'r2':r2
    }
    return error_metrics
