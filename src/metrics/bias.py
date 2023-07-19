import numpy as np


def normalized_bias_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized forecast bias.

    Metric values will stay between -1 and 1, with 0 indicating the
    absence of bias. Consistent negative values indicate a tendency
    to under-forecast whereas constant positive values indicate a
    tendency to over-forecast.

    Parameters
    ----------
    y_true : array-like
        Ground truth
    y_pred : array-like
        Estimated target values.

    Returns
    -------
    bias : float
        Metric value
    """
    num = y_pred - y_true
    den = y_pred + y_true
    return np.sum(np.divide(num, den))
