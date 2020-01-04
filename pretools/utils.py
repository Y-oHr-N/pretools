"""Utilities."""

import numpy as np
import pandas as pd


def get_categorical_cols(X: pd.DataFrame, labels: bool = False) -> pd.Series:
    """Get categorical columns."""
    X = pd.DataFrame(X)
    is_categorical = X.dtypes == 'category'

    if labels:
        return X.columns[is_categorical]

    return is_categorical


def get_numerical_cols(X: pd.DataFrame, labels: bool = False) -> pd.Series:
    """Get numerical columns."""
    X = pd.DataFrame(X)
    is_numerical = X.dtypes.apply(lambda x: issubclass(x.type, np.number))

    if labels:
        return X.columns[is_numerical]

    return is_numerical


def get_time_cols(X: pd.DataFrame, labels: bool = False) -> pd.Series:
    """Get time columns."""
    X = pd.DataFrame(X)

    is_time = X.dtypes.apply(lambda x: issubclass(x.type, np.datetime64))

    if labels:
        return X.columns[is_time]

    return is_time
