import numpy as np

from src.metrics.intervals import coverage_fraction
from src.uncertainty.intervals import (
    calculate_prediction_intervals,
    get_residuals,
    get_residuals_quantile,
)


def cross_validation_predict(estimators, X, y, cv) -> dict:
    """Generate cross-validated estimates for each input data point.

    Parameters
    ----------
    estimators : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target variable to try to predict in the case of supervised learning.

    cv : cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.

    Returns
    -------
    predictions : dict of lists of float arrays
        Array of predictions of the estimator for each run of the cross validation.
    """
    dates = []
    actual_values = []
    predictions = []
    residuals = []
    for estimator, fold in zip(estimators, cv.split(X, y)):
        _, valid_idx = fold
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx].values

        valid_pred = estimator.predict(X_valid)
        valid_resid = get_residuals(y_valid, valid_pred)

        # Why not return a pd.DataFrame? O a DataClass?
        # Consistency with scikit-learn (lists of arrays)
        dates.append(X_valid.index)
        actual_values.append(y_valid)
        predictions.append(valid_pred)
        residuals.append(valid_resid)

    return {
        "test_dates": dates,
        "test_actuals": actual_values,
        "test_point_estimates": predictions,
        "test_residuals": residuals,
    }


def cross_validation_intervals(
    cv_predictions: dict[str, np.ndarray],
    std_multiplier: float = 1.96,
) -> dict:
    """Compute cross validation confidence intervals for each fold."""
    dates = cv_predictions["test_dates"]
    actuals = cv_predictions["test_actuals"]
    residuals = cv_predictions["test_residuals"]
    point_estimates = cv_predictions["test_point_estimates"]
    forecasts = []
    for date, actual, resid, pred in zip(dates, actuals, residuals, point_estimates):
        ci = get_residuals_quantile(resid, std_multiplier)
        pred_df = calculate_prediction_intervals(pred, ci)
        pred_df.set_index(date, inplace=True)
        pred_df.insert(0, "y_true", actual)
        forecasts.append(pred_df)

    cv_predictions.update({"test_predictions": forecasts})
    return cv_predictions


def cross_validation_coverage(predictions) -> np.ndarray:
    """For each cross validation fold compute the coverage metric."""
    scores = [
        coverage_fraction(p.y_true, p.y_pred_lower, p.y_pred_upper)
        for p in predictions["test_predictions"]
    ]
    return np.asarray(scores)
