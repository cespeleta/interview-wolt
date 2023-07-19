import numpy as np


def coverage_fraction(
    y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray
) -> float:
    """Fraction of observations that fall between the intervals.

    Parameters
    ----------
    y_true : array-like
        Ground truth
    y_low : array-like
        Estimated low confidence interval values.
    y_high : array-like
        Estimated high confidence interval values.

    Returns
    -------
    coverage : float
        Metric value
    """
    return np.mean(np.logical_and(y_true >= y_low, y_true <= y_high))
