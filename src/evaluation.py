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
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    mape = 100 * (abs(actual - predicted) / actual).mean()
    r2 = r2_score(actual, predicted)
    error_metrics = {
            'mae':mae,
            'mse':mse,
            'rmse':rmse,
            'mape':mape,
            'r2':r2
    }
    return error_metrics
