from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_model(y_true, y_pred):
    """Calculate basic regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse,
    }