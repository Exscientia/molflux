import numpy as np
import scipy.stats as st

from molflux.metrics.typing import ArrayLike


def _estimate_standard_deviation(
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
    confidence: float,
) -> ArrayLike:
    # solve for standard deviation sigma based on difference between cdf at upper and lower bound
    # simpler estimate ignoring mu, and assuming symmetric gaussian distns for predictions
    def diff(x: ArrayLike) -> ArrayLike:
        return x[1] - x[0]

    prediction_intervals = list(zip(lower_bound, upper_bound))
    sigma = [
        diff(interval) / diff(st.norm.interval(confidence))
        for interval in prediction_intervals
    ]
    return np.array(sigma)
