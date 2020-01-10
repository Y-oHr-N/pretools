"""Estimators."""

import itertools
import logging

from typing import Any
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

try:  # scikit-learn<=0.21
    from sklearn.feature_selection.from_model import _calculate_threshold
    from sklearn.feature_selection.from_model import _get_feature_importances
except ImportError:
    from sklearn.feature_selection._from_model import _calculate_threshold
    from sklearn.feature_selection._from_model import _get_feature_importances

from .utils import get_numerical_cols


class CalendarFeatures(BaseEstimator, TransformerMixin):
    """Calendar features."""

    def __init__(
        self,
        dtype: Union[str, Type] = 'float64',
        encode: bool = False,
        include_unixtime: bool = False
    ) -> None:
        self.dtype = dtype
        self.encode = encode
        self.include_unixtime = include_unixtime

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'CalendarFeatures':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        X = pd.DataFrame(X)

        secondsinminute = 60.0
        secondsinhour = 60.0 * secondsinminute
        secondsinday = 24.0 * secondsinhour
        secondsinweekday = 7.0 * secondsinday
        secondsinmonth = 30.4167 * secondsinday
        secondsinyear = 12.0 * secondsinmonth

        self.attributes_ = {}

        for col in X:
            s = X[col]
            duration = s.max() - s.min()
            duration = duration.total_seconds()
            attrs = []

            if duration >= 2.0 * secondsinyear:
                # if s.dt.dayofyear.nunique() > 1:
                #     attrs.append('dayofyear')
                # if s.dt.weekofyear.nunique() > 1:
                #     attrs.append('weekofyear')
                # if s.dt.quarter.nunique() > 1:
                #     attrs.append('quarter')
                if s.dt.month.nunique() > 1:
                    attrs.append('month')
            if duration >= 2.0 * secondsinmonth \
                    and s.dt.day.nunique() > 1:
                attrs.append('day')
            if duration >= 2.0 * secondsinweekday \
                    and s.dt.weekday.nunique() > 1:
                attrs.append('weekday')
            if duration >= 2.0 * secondsinday \
                    and s.dt.hour.nunique() > 1:
                attrs.append('hour')
            # if duration >= 2.0 * secondsinhour \
            #         and s.dt.minute.nunique() > 1:
            #     attrs.append('minute')
            # if duration >= 2.0 * secondsinminute \
            #         and s.dt.second.nunique() > 1:
            #     attrs.append('second')

            self.attributes_[col] = attrs

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        for col in X:
            s = X[col]

            if self.include_unixtime:
                unixtime = 1e-09 * s.astype('int64')
                unixtime = unixtime.astype(self.dtype)

                Xt[col] = unixtime

            for attr in self.attributes_[col]:
                x = getattr(s.dt, attr)

                if not self.encode:
                    x = x.astype('category')

                    Xt['{}_{}'.format(col, attr)] = x

                    continue

                # if attr == 'dayofyear':
                #     period = np.where(s.dt.is_leap_year, 366.0, 365.0)
                # elif attr == 'weekofyear':
                #     period = 52.1429
                # elif attr == 'quarter':
                #     period = 4.0
                elif attr == 'month':
                    period = 12.0
                elif attr == 'day':
                    period = s.dt.daysinmonth
                elif attr == 'weekday':
                    period = 7.0
                elif attr == 'hour':
                    x += s.dt.minute / 60.0 + s.dt.second / 60.0
                    period = 24.0
                # elif attr in ['minute', 'second']:
                #     period = 60.0

                theta = 2.0 * np.pi * x / period
                sin_theta = np.sin(theta)
                sin_theta = sin_theta.astype(self.dtype)
                cos_theta = np.cos(theta)
                cos_theta = cos_theta.astype(self.dtype)

                Xt['{}_{}_sin'.format(col, attr)] = sin_theta
                Xt['{}_{}_cos'.format(col, attr)] = cos_theta

        return Xt


class ClippedFeatures(BaseEstimator, TransformerMixin):
    """Clipped features."""

    def __init__(self, high: float = 0.99, low: float = 0.01) -> None:
        self.high = high
        self.low = low

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'ClippedFeatures':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        X = pd.DataFrame(X)

        self.data_max_ = X.quantile(q=self.high)
        self.data_min_ = X.quantile(q=self.low)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)

        return X.clip(self.data_min_, self.data_max_, axis=1)


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """Combined Features."""

    def __init__(
        self,
        include_data: bool = False,
        max_features: Optional[int] = None,
        operands: Optional[List[str]] = None
    ) -> None:
        self.include_data = include_data
        self.max_features = max_features
        self.operands = operands

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'CombinedFeatures':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()
        numerical_cols = get_numerical_cols(X, labels=True)
        other_cols = np.setdiff1d(X.columns, numerical_cols)

        n_features = 0

        if self.max_features is None:
            max_features = np.inf
        else:
            max_features = self.max_features

        if self.operands is None:
            operands = [
                'add',
                'subtract',
                'multiply',
                'divide',
                # 'equal'
            ]
        else:
            operands = self.operands

        for col1, col2 in itertools.combinations(other_cols, 2):
            if n_features >= max_features:
                break

            func = np.vectorize(lambda x1, x2: '{}+{}'.format(x1, x2))

            feature = func(X[col1], X[col2])
            feature = pd.Series(feature, index=X.index)
            Xt['add_{}_{}'.format(col1, col2)] = feature.astype('category')

            n_features += 1

        for col1, col2 in itertools.combinations(numerical_cols, 2):
            for operand in operands:
                if n_features >= max_features:
                    break

                func = getattr(np, operand)

                Xt['{}_{}_{}'.format(operand, col1, col2)] = func(
                    X[col2], X[col2]
                )

                n_features += 1

        if self.include_data:
            Xt = pd.concat([X, Xt], axis=1)

        return Xt


class DiffFeatures(BaseEstimator, TransformerMixin):
    """Diff features."""

    def __init__(self, include_data: bool = False) -> None:
        self.include_data = include_data

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'DiffFeatures':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)
        Xt = X.diff()
        Xt = Xt.rename(columns='{}_diff'.format)

        if self.include_data:
            Xt = pd.concat([X, Xt], axis=1)

        return Xt


class DropCollinearFeatures(BaseEstimator, TransformerMixin):
    """Feature selector that removes collinear features."""

    def __init__(
        self,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        subsample: Union[int, float] = 0.75,
        threshold: float = 0.95
    ) -> None:
        self.random_state = random_state
        self.subsample = subsample
        self.threshold = threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'DropCollinearFeatures':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        X = pd.DataFrame(X)

        X, _, = train_test_split(
            X,
            random_state=self.random_state,
            train_size=self.subsample
        )

        self.corr_ = X.corr()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)

        triu = np.triu(self.corr_, k=1)
        triu = np.abs(triu)
        triu = np.nan_to_num(triu)

        logger = logging.getLogger(__name__)

        cols = np.all(triu <= self.threshold, axis=0)
        _, n_features = X.shape
        n_dropped_features = n_features - np.sum(cols)

        logger.info('{} features are dropped.'.format(n_dropped_features))

        return X.loc[:, cols]


class ModifiedSelectFromModel(BaseEstimator, TransformerMixin):
    """Meta-transformer for selecting features based on importance weights."""

    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        subsample: Union[int, float] = 0.75,
        threshold: Optional[Union[float, str]] = None
    ) -> None:
        self.estimator = estimator
        self.random_state = random_state
        self.subsample = subsample
        self.threshold = threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **fit_params: Any
    ) -> 'ModifiedSelectFromModel':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        X, _, y, _ = train_test_split(
            X,
            y,
            random_state=self.random_state,
            train_size=self.subsample
        )

        self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y, **fit_params)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)

        feature_importances = _get_feature_importances(self.estimator_)
        threshold = _calculate_threshold(
            self.estimator_, feature_importances, self.threshold
        )

        logger = logging.getLogger(__name__)

        cols = feature_importances >= threshold
        _, n_features = X.shape
        n_dropped_features = n_features - np.sum(cols)

        logger.info('{} features are dropped.'.format(n_dropped_features))

        return X.loc[:, cols]


class NAValuesThreshold(BaseEstimator, TransformerMixin):
    """Feature selector that removes features with many missing values."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'NAValuesThreshold':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        X = pd.DataFrame(X)

        self.n_samples_, _ = X.shape
        self.count_ = X.count()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)

        logger = logging.getLogger(__name__)

        cols = self.count_ >= (1.0 - self.threshold) * self.n_samples_
        _, n_features = X.shape
        n_dropped_features = n_features - np.sum(cols)

        logger.info('{} features are dropped.'.format(n_dropped_features))

        return X.loc[:, cols]


class NUniqueThreshold(BaseEstimator, TransformerMixin):
    """Feature selector that removes low and high cardinal features."""

    def __init__(
        self,
        max_freq: Optional[Union[float, int]] = 1.0,
        min_freq: Union[float, int] = 1
    ) -> None:
        self.max_freq = max_freq
        self.min_freq = min_freq

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'NUniqueThreshold':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        X = pd.DataFrame(X)

        self.n_samples_, _ = X.shape
        self.nunique_ = X.nunique()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)

        if self.max_freq is None:
            max_freq = np.inf
        elif isinstance(self.max_freq, float):
            max_freq = int(self.max_freq * self.n_samples_)
        else:
            max_freq = self.max_freq

        if isinstance(self.min_freq, float):
            min_freq = int(self.min_freq * self.n_samples_)
        else:
            min_freq = self.min_freq

        logger = logging.getLogger(__name__)

        cols = (self.nunique_ > min_freq) & (self.nunique_ < max_freq)
        _, n_features = X.shape
        n_dropped_features = n_features - np.sum(cols)

        logger.info('{} features are dropped.'.format(n_dropped_features))

        return X.loc[:, cols]


class Profiler(BaseEstimator, TransformerMixin):
    """Profiler."""

    def __init__(
        self,
        label_col: str = 'label',
        max_columns: Optional[int] = None,
        precision: int = 3
    ) -> None:
        self.label_col = label_col
        self.max_columns = max_columns
        self.precision = precision

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'Profiler':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        data = pd.DataFrame(X)

        if y is not None:
            kwargs = {self.label_col: y}
            data = X.assign(**kwargs)

        logger = logging.getLogger(__name__)
        summary = data.describe(include='all')

        with pd.option_context(
            'display.max_columns',
            self.max_columns,
            'display.precision',
            self.precision
        ):
            logger.info(summary)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        return pd.DataFrame(X)


class RowStatistics(BaseEstimator, TransformerMixin):
    """Row statistics."""

    def __init__(self, dtype: Union[str, Type] = 'float64') -> None:
        self.dtype = dtype

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'RowStatistics':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        is_null = X.isnull()

        Xt['number_of_na_values'] = is_null.sum(axis=1)

        return Xt.astype(self.dtype)


class TextStatistics(BaseEstimator, TransformerMixin):
    """Text statistics."""

    def __init__(self, dtype: Union[str, Type] = 'float64') -> None:
        self.dtype = dtype

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'TextStatistics':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        for col in X:
            Xt['{}_len'.format(col)] = X[col].str.len()

        return Xt
