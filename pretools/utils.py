"""Utilities."""

import logging

import numpy as np
import pandas as pd


def check_X(X: pd.DataFrame) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X

    return pd.DataFrame(X)


def get_categorical_cols(X: pd.DataFrame, labels: bool = False) -> pd.Series:
    """Get categorical columns."""
    X = pd.DataFrame(X)
    logger = logging.getLogger(__name__)
    is_categorical = X.dtypes == "category"
    n_features = np.sum(is_categorical)

    logger.info("The number of categorical features is {}.".format(n_features))

    if labels:
        return X.columns[is_categorical]

    return is_categorical


def get_numerical_cols(X: pd.DataFrame, labels: bool = False) -> pd.Series:
    """Get numerical columns."""
    X = pd.DataFrame(X)
    logger = logging.getLogger(__name__)
    is_numerical = X.dtypes.apply(lambda x: issubclass(x.type, np.number))
    n_features = np.sum(is_numerical)

    logger.info("The number of numerical features is {}.".format(n_features))

    if labels:
        return X.columns[is_numerical]

    return is_numerical


def get_time_cols(X: pd.DataFrame, labels: bool = False) -> pd.Series:
    """Get time columns."""
    X = pd.DataFrame(X)
    logger = logging.getLogger(__name__)
    is_time = X.dtypes.apply(lambda x: issubclass(x.type, np.datetime64))
    n_features = np.sum(is_time)

    logger.info("The number of time features is {}.".format(n_features))

    if labels:
        return X.columns[is_time]

    return is_time


def get_unknown_cols(X: pd.DataFrame, labels: bool = False) -> pd.Series:
    """Get unknown columns."""
    X = pd.DataFrame(X)
    logger = logging.getLogger(__name__)
    is_unknown = X.dtypes == object
    n_features = np.sum(is_unknown)

    logger.info("The number of unknown features is {}.".format(n_features))

    if labels:
        return X.columns[is_unknown]

    return is_unknown
