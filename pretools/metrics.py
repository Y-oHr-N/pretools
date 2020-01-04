"""Metrics."""

from typing import Optional
from typing import Union

import numpy as np

try:  # scikit-learn<=0.21
    from sklearn.metrics.regression import _check_reg_targets
except ImportError:
    from sklearn.metrics._regression import _check_reg_targets


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = 'uniform_average'
) -> Union[float, np.ndarray]:
    """Mean Absolute Percentage Error (MAPE)."""
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true,
        y_pred,
        multioutput
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true - y_pred) / y_true)

    is_nan = np.isnan(ape)
    ape[is_nan] = 0.0
    mape = np.average(ape, axis=0, weights=sample_weight)

    if multioutput == 'raw_values':
        return mape
    elif multioutput == 'uniform_average':
        multioutput = None  # type: ignore

    return np.average(mape, weights=multioutput)


def mean_arctangent_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = 'uniform_average'
) -> Union[float, np.ndarray]:
    """Mean Arctangent Absolute Percentage Error (MAAPE)."""
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true,
        y_pred,
        multioutput
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true - y_pred) / y_true)

    is_nan = np.isnan(ape)
    ape[is_nan] = 0.0
    aape = np.arctan(ape)
    maape = np.average(aape, axis=0, weights=sample_weight)

    if multioutput == 'raw_values':
        return maape
    elif multioutput == 'uniform_average':
        multioutput = None  # type: ignore

    return np.average(maape, weights=multioutput)
