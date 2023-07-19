import numpy as np
import pandas as pd


def get_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals.

    Residuals are computed as y_true - y_pred.

    Parameters
    ----------
    y_true : array-like
        Ground truth

    y_pred : array-like
        Estimated target values.

    Returns
    -------
    residuals : array-like
        Contains model residuals.
    """
    return y_true - y_pred


def get_residuals_quantile(
    residuals: np.ndarray,
    std_multiplier: float = 1.96,
    n_boot: int = 100,
) -> float:
    """Calculate quantile of the input residuals.

    Parameters
    ----------
    residuals : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Model residuals

    std_multiplier : float, default=1.96
        Standard deviation multiplier.

    n_boot : int, default=100
        Number of boostrap samples

    Returns
    -------
    Dispersion : float
        Dispersion value
    """
    size = len(residuals) // 3
    quantiles = [
        np.median(np.random.choice(residuals, size=size)) for _ in range(n_boot)
    ]
    return np.std(quantiles) * std_multiplier


# predict_confidence_intervals?
def calculate_prediction_intervals(
    y_pred: np.ndarray, confidence_interval: float, time_interval: bool = True
) -> pd.DataFrame:
    """Generate prediction intervals.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    confidence_interval : float
        Error measured in the residuals.

    time_interval : bool
        Further away points have greater uncertainty.

    Returns
    -------
    predictions : DataFrame
        Contains predicted values and confidence intervals.
    """
    if time_interval:
        index = range(1, len(y_pred) + 1)
    else:
        index = np.ones(shape=len(y_pred))  # [1] * len(y_pred)

    return pd.DataFrame(
        {
            "y_pred": y_pred,
            "y_pred_lower": y_pred - confidence_interval * np.sqrt(index),
            "y_pred_upper": y_pred + confidence_interval * np.sqrt(index),
        }
    )
