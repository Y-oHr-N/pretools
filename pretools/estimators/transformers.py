"""Transformers."""

from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


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
