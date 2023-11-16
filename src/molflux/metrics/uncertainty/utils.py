from typing import cast

import numpy as np
import scipy.stats as st
from scipy.optimize import fsolve
from sklearn.utils.validation import column_or_1d

from molflux.metrics.typing import ArrayLike


def _estimate_standard_deviation(
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
    mu: ArrayLike,
    confidence: float,
) -> ArrayLike:
    # solve for standard deviation sigma based on difference between cdf at upper and lower bound
    # centered on predictions which should equate to confidence
    def zero_func(scale: float) -> float:
        return (  # type: ignore[no-any-return]
            st.norm.cdf(upper_bound, loc=mu, scale=scale)
            - st.norm.cdf(lower_bound, loc=mu, scale=scale)
            - confidence
        )

    interval_width = np.array(upper_bound) - np.array(lower_bound)
    estimated_width_in_standard_deviations = st.norm.ppf(
        1 - (1 - confidence) / 2,
    ) - st.norm.ppf((1 - confidence) / 2)
    sigma = fsolve(
        zero_func,
        x0=interval_width / estimated_width_in_standard_deviations,
    )
    return sigma


def regression_coverage_score(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike,
) -> float:
    """
    Effective coverage score obtained by the prediction intervals.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction intervals.

    Vendored from https://github.com/scikit-learn-contrib/MAPIE

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True labels.
    y_pred_low : ArrayLike of shape (n_samples,)
        Lower bound of prediction intervals.
    y_pred_up : ArrayLike of shape (n_samples,)
        Upper bound of prediction intervals.

    Returns
    -------
    float
        Effective coverage obtained by the prediction intervals.

    Examples
    --------
    >>> from mapie.metrics import regression_coverage_score
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_pred_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_pred_up = np.array([6, 9, 10, 12.5, 12])
    >>> print(regression_coverage_score(y_true, y_pred_low, y_pred_up))
    0.8
    """
    y_true = cast(np.ndarray, column_or_1d(y_true))
    y_pred_low = cast(np.ndarray, column_or_1d(y_pred_low))
    y_pred_up = cast(np.ndarray, column_or_1d(y_pred_up))
    coverage = np.mean((y_pred_low <= y_true) & (y_pred_up >= y_true))
    return float(coverage)
