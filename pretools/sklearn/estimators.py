"""Estimators."""

import itertools
import logging

from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import sklearn

from scipy.sparse import issparse
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.model_selection import train_test_split
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from tqdm import trange

if sklearn.__version__ >= "0.22":
    from sklearn.feature_selection._from_model import _calculate_threshold
    from sklearn.feature_selection._from_model import _get_feature_importances
    from sklearn.utils import _safe_indexing  # noqa
else:
    from sklearn.feature_selection.from_model import _calculate_threshold
    from sklearn.feature_selection.from_model import _get_feature_importances
    from sklearn.utils import safe_indexing as _safe_indexing  # noqa

from ..utils import check_X
from ..utils import get_categorical_cols
from ..utils import get_numerical_cols
from ..utils import get_time_cols
from ..utils import get_unknown_cols
from ..utils import sigmoid

MAX_INT = np.iinfo(np.int32).max


def make_modified_column_transformer(
    *transformers: Tuple,
) -> "ModifiedColumnTransformer":
    """Make ModifedColumnTransformer.

    Examples
    --------
    >>> from pretools.sklearn.estimators import make_modified_column_transformer  # noqa
    >>> transformers = [("passthrough", [0])]
    >>> est = make_modified_column_transformer(*transformers)
    """
    transformer_list = _get_transformer_list(transformers)

    return ModifiedColumnTransformer(transformer_list)


class SelectorMixin(TransformerMixin):
    """Mixin class for all selectors."""

    @abstractmethod
    def get_support(self) -> np.ndarray:
        """Get a mask of the features selected.

        Returns
        -------
        mask
            Boolean mask.
        """

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
        logger = logging.getLogger(__name__)

        X = check_X(X, estimator=self, force_all_finite="allow-nan")
        mask = self.get_support()

        logger.info(
            "{} dropped {} features.".format(
                self.__class__.__name__, np.sum(~mask)
            )
        )

        return X.loc[:, mask]


class Astype(BaseEstimator, TransformerMixin):
    """Astype.

    Examples
    --------
    >>> from pretools.sklearn.estimators import Astype
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> est = Astype()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(self, copy: bool = True) -> None:
        self.copy = copy

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Astype":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        if self.copy:
            X = X.copy()

        numerical_cols = get_numerical_cols(X, labels=True)
        unknown_cols = get_unknown_cols(X, labels=True)

        if len(numerical_cols) > 0:
            X[numerical_cols] = X[numerical_cols].astype("float32")

        if len(unknown_cols) > 0:
            X[unknown_cols] = X[unknown_cols].astype("category")

        return X


class CalendarFeatures(BaseEstimator, TransformerMixin):
    """Calendar features.

    Examples
    --------
    >>> import datetime
    >>> from pretools.sklearn.estimators import CalendarFeatures
    >>> X = [
    ...     [datetime.datetime(2000, 1, 1, 0, 0, 0)],
    ...     [np.nan],
    ...     [datetime.datetime(2010, 10, 10, 10, 0, 0)]
    ... ]
    >>> est = CalendarFeatures()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(
        self,
        dtype: Union[str, Type] = "float64",
        encode: bool = False,
        include_unixtime: bool = False,
    ) -> None:
        self.dtype = dtype
        self.encode = encode
        self.include_unixtime = include_unixtime

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CalendarFeatures":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

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
                #     attrs.append("dayofyear")
                # if s.dt.weekofyear.nunique() > 1:
                #     attrs.append("weekofyear")
                # if s.dt.quarter.nunique() > 1:
                #     attrs.append("quarter")
                if s.dt.month.nunique() > 1:
                    attrs.append("month")
            if duration >= 2.0 * secondsinmonth and s.dt.day.nunique() > 1:
                attrs.append("day")
            if (
                duration >= 2.0 * secondsinweekday
                and s.dt.weekday.nunique() > 1
            ):
                attrs.append("weekday")
            if duration >= 2.0 * secondsinday and s.dt.hour.nunique() > 1:
                attrs.append("hour")
            # if (
            #     duration >= 2.0 * secondsinhour
            #     and s.dt.minute.nunique() > 1
            # ):
            #     attrs.append("minute")
            # if (
            #     duration >= 2.0 * secondsinminute
            #     and s.dt.second.nunique() > 1
            # ):
            #     attrs.append("second")

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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        Xt = pd.DataFrame()

        for col in X:
            s = X[col]

            if self.include_unixtime:
                unixtime = 1e-09 * s.astype("int64")
                unixtime = unixtime.astype(self.dtype)

                Xt["{}_unixtime".format(col)] = unixtime

            for attr in self.attributes_[col]:
                x = getattr(s.dt, attr)

                if not self.encode:
                    x = x.astype("category")

                    Xt["{}_{}".format(col, attr)] = x

                    continue

                # if attr == "dayofyear":
                #     period = np.where(s.dt.is_leap_year, 366.0, 365.0)
                # elif attr == "weekofyear":
                #     period = 52.1429
                # elif attr == "quarter":
                #     period = 4.0
                elif attr == "month":
                    period = 12.0
                elif attr == "day":
                    period = s.dt.daysinmonth
                elif attr == "weekday":
                    period = 7.0
                elif attr == "hour":
                    x += s.dt.minute / 60.0 + s.dt.second / 60.0
                    period = 24.0
                # elif attr in ["minute", "second"]:
                #     period = 60.0

                theta = 2.0 * np.pi * x / period
                sin_theta = np.sin(theta)
                sin_theta = sin_theta.astype(self.dtype)
                cos_theta = np.cos(theta)
                cos_theta = cos_theta.astype(self.dtype)

                Xt["{}_{}_sin".format(col, attr)] = sin_theta
                Xt["{}_{}_cos".format(col, attr)] = cos_theta

        logger = logging.getLogger(__name__)

        _, n_created_features = Xt.shape

        logger.info(
            "{} created {} features.".format(
                self.__class__.__name__, n_created_features
            )
        )

        return Xt


class ClippedFeatures(BaseEstimator, TransformerMixin):
    """Clipped features.

    Examples
    --------
    >>> from pretools.sklearn.estimators import ClippedFeatures
    >>> X = [[10, np.nan, 4], [0, 2, 1]]
    >>> est = ClippedFeatures()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(
        self, copy: bool = True, high: float = 0.99, low: float = 0.01,
    ) -> None:
        self.copy = copy
        self.high = high
        self.low = low

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ClippedFeatures":
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
        X = check_X(X, estimator=self, force_all_finite="allow-nan")

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
        X = check_X(X, estimator=self, force_all_finite="allow-nan")

        if self.copy:
            X = X.copy()

        X.clip(self.data_min_, self.data_max_, axis=1, inplace=True)

        return X


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """Combined Features.

    Examples
    --------
    >>> from pretools.sklearn.estimators import CombinedFeatures
    >>> X = [[1, 1], [1, 2], [1, np.nan], [1, -10]]
    >>> est = CombinedFeatures()
    >>> Xt = est.fit_transform(X)
    """

    @property
    def _operands(self) -> List[str]:
        if self.operands is None:
            return [
                "add",
                "subtract",
                "multiply",
                "divide",
                # "equal",
            ]

        return self.operands

    def __init__(
        self,
        include_data: bool = False,
        max_features: Optional[Union[int, str]] = "auto",
        operands: Optional[List[str]] = None,
    ) -> None:
        self.include_data = include_data
        self.max_features = max_features
        self.operands = operands

    def _numerical_transform(
        self, X: pd.DataFrame, max_features: int,
    ) -> pd.DataFrame:
        Xt = pd.DataFrame()
        n_features = 0

        for col1, col2 in itertools.combinations(X.columns, 2):
            for operand in self._operands:
                if n_features >= max_features:
                    break

                func = getattr(np, operand)

                Xt["{}_{}_{}".format(operand, col1, col2)] = func(
                    X[col1], X[col2]
                )

                n_features += 1

        return Xt

    def _other_transform(
        self, X: pd.DataFrame, max_features: int,
    ) -> pd.DataFrame:
        Xt = pd.DataFrame()
        n_features = 0

        for col1, col2 in itertools.combinations(X.columns, 2):
            for operand in self._operands:
                if n_features >= max_features:
                    break

                if operand == "multiply":
                    func = np.vectorize(lambda x1, x2: "{}*{}".format(x1, x2))
                elif operand == "equal":
                    func = np.equal
                else:
                    continue

                try:
                    feature = func(X[col1], X[col2])
                except TypeError:
                    continue

                if operand == "multiply":
                    feature = pd.Series(feature, index=X.index)
                    feature = feature.astype("category")

                Xt["{}_{}_{}".format(operand, col1, col2)] = feature

                n_features += 1

        return Xt

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CombinedFeatures":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        self.n_samples_, self.n_features_ = X.shape

        if self.max_features is None:
            self.max_features_ = np.inf
        elif self.max_features == "auto":
            if self.include_data:
                self.max_features_ = self.n_samples_ - self.n_features_ - 1
            else:
                self.max_features_ = self.n_samples_ - 1
        else:
            self.max_features_ = self.max_features

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
        X = check_X(X, dtype=None, force_all_finite="allow-nan")
        is_numerical = get_numerical_cols(X)

        logger = logging.getLogger(__name__)

        Xt_numerical = self._numerical_transform(
            X.loc[:, is_numerical], self.max_features_,
        )
        _, n_created_features = Xt_numerical.shape

        Xt_other = self._other_transform(
            X.loc[:, ~is_numerical], self.max_features_ - n_created_features,
        )

        Xt = pd.concat([Xt_numerical, Xt_other], axis=1)
        _, n_created_features = Xt.shape

        logger.info(
            "{} created {} features.".format(
                self.__class__.__name__, n_created_features
            )
        )

        if self.include_data:
            Xt = pd.concat([X, Xt], axis=1)

        return Xt


class DiffFeatures(BaseEstimator, TransformerMixin):
    """Diff features.

    Examples
    --------
    >>> import numpy as np
    >>> from pretools.sklearn.estimators import DiffFeatures
    >>> est = DiffFeatures()
    >>> X = [[1], [np.nan], [1], [10], [1]]
    >>> Xt = est.fit_transform(X)
    """

    def __init__(self, include_data: bool = False) -> None:
        self.include_data = include_data

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DiffFeatures":
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
        X = check_X(X, estimator=self, force_all_finite="allow-nan")
        Xt = X.diff()

        Xt.rename(columns="{}_diff".format, inplace=True)

        if self.include_data:
            Xt = pd.concat([X, Xt], axis=1)

        return Xt


class DropCollinearFeatures(BaseEstimator, SelectorMixin):
    """Feature selector that removes collinear features.

    Examples
    --------
    >>> from pretools.sklearn.estimators import DropCollinearFeatures
    >>> X = [[1, 1, 1], [2, 2, 200], [3, 3, 3000], [1, np.nan, 1]]
    >>> est = DropCollinearFeatures()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(
        self,
        method: Union[Callable, str] = "pearson",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: bool = True,
        subsample: Union[int, float] = 0.75,
        threshold: float = 0.95,
    ) -> None:
        self.method = method
        self.random_state = random_state
        self.shuffle = shuffle
        self.subsample = subsample
        self.threshold = threshold

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DropCollinearFeatures":
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
        X = check_X(X, estimator=self, force_all_finite="allow-nan")

        if self.subsample < 1.0:
            X, _, = train_test_split(
                X,
                random_state=self.random_state,
                train_size=self.subsample,
                shuffle=self.shuffle,
            )

        self.corr_ = X.corr(method=self.method)

        return self

    def get_support(self) -> np.ndarray:
        """Get a mask of the features selected.

        Returns
        -------
        mask
            Boolean mask.
        """
        triu = np.triu(self.corr_, k=1)
        triu = np.abs(triu)
        triu = np.nan_to_num(triu)

        return np.all(triu <= self.threshold, axis=0)


class DropDriftFeatures(BaseEstimator, SelectorMixin):
    """Drop drift features.

    Examples
    --------
    >>> import numpy as np
    >>> from pretools.sklearn.estimators import DropDriftFeatures
    >>> sel = DropDriftFeatures()
    >>> X = [[1, 1, 100], [2, 2, 10], [1, 1, 1], [np.nan, 1, 1]]
    >>> X_test = [[1, 1000, 100], [2, 300, 10], [1, 100, 1], [1, 100, 1]]
    >>> Xt = sel.fit_transform(X, X_test=X_test)
    >>> Xt.shape
    (4, 2)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_samples: int = 100_000,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        self.alpha = alpha
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> "DropDriftFeatures":
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        X_test
            Test data.

        Returns
        -------
        self
            Return self.
        """
        if X_test is None:
            raise ValueError("X_test must be given.")

        X = check_X(X, estimator=self, force_all_finite="allow-nan")
        X_test = check_X(X_test, estimator=self, force_all_finite="allow-nan")
        random_state = check_random_state(self.random_state)
        train_size, _ = X.shape
        train_size = min(train_size, self.max_samples)
        test_size, _ = X_test.shape
        test_size = min(test_size, self.max_samples)

        _, self.n_features_ = X.shape
        self.pvalues_ = np.empty(self.n_features_)

        for j, column in enumerate(X):
            x = X[column]
            x_test = X_test[column]
            is_nan = pd.isnull(x)
            is_nan_test = pd.isnull(x_test)
            train = np.where(~is_nan)[0]
            train = random_state.choice(train, size=train_size)
            test = np.where(~is_nan_test)[0]
            test = random_state.choice(test, size=test_size)
            x = _safe_indexing(x, train)
            x_test = _safe_indexing(x_test, test)

            if issparse(x):
                x = np.ravel(x.toarray())

            if issparse(x_test):
                x_test = np.ravel(x_test.toarray())

            self.pvalues_[j] = ks_2samp(x, x_test).pvalue

        return self

    def get_support(self) -> np.ndarray:
        """Get a mask of the features selected.

        Returns
        -------
        mask
            Boolean mask.
        """
        return self.pvalues_ >= self.alpha


class ModifiedCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """Modified CatBoostClassifier.

    Examples
    --------
    >>> import pandas as pd
    >>> from pretools.sklearn.estimators import ModifiedCatBoostClassifier
    >>> X = [["Cat"], ["Cow"], ["Mouse"], ["Lion"]]
    >>> X = pd.DataFrame(X)
    >>> X = X.astype("category")
    >>> y = [0, 1, 1, 0]
    >>> est = ModifiedCatBoostClassifier(verbose=0)
    >>> est.fit(X, y)
    ModifiedCatBoostClassifier(...)
    >>> y_pred = est.predict(X)
    """

    @property
    def classes_(self) -> np.ndarray:
        """Class labels."""
        return self._encoder.classes_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances."""
        return self._model.get_feature_importance()

    @property
    def predict_proba(self) -> Callable[[np.ndarray], np.ndarray]:
        """Predict class probabilities for data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        p
            Class probabilities of data.
        """
        return self._model.predict_proba

    def __init__(self, **params: Any) -> None:
        from catboost import CatBoostClassifier

        self._params = params
        self._encoder = LabelEncoder()
        self._model = CatBoostClassifier(**params)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params
            Estimator parameters.
        """
        params = self._model.get_params(deep=deep)

        params.update(self._params)

        return params

    def set_params(self, **params: Any) -> "ModifiedCatBoostClassifier":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self
            Return self.
        """
        for key, value in params.items():
            self._params[key] = value

        self._model.set_params(**params)

        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray, **fit_params: Any
    ) -> "ModifiedCatBoostClassifier":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        y = self._encoder.fit_transform(y)

        if "cat_features" not in fit_params:
            cat_features = get_categorical_cols(X, labels=True)
            fit_params["cat_features"] = cat_features

        self._model.fit(X, y, **fit_params)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """
        y_pred = self._model.predict(X)
        y_pred = np.ravel(y_pred)
        y_pred = y_pred.astype("int64")

        return self._encoder.inverse_transform(y_pred)


class ModifiedCatBoostRegressor(BaseEstimator, RegressorMixin):
    """Modified CatBoostRegressor.

    Examples
    --------
    >>> import pandas as pd
    >>> from pretools.sklearn.estimators import ModifiedCatBoostRegressor
    >>> X = [["Cat"], ["Cow"], ["Mouse"], ["Lion"]]
    >>> X = pd.DataFrame(X)
    >>> X = X.astype("category")
    >>> y = [0.0, 1.0, 2.0, 0.0]
    >>> est = ModifiedCatBoostRegressor(verbose=0)
    >>> est.fit(X, y)
    ModifiedCatBoostRegressor(...)
    >>> y_pred = est.fit(X, y)
    """

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances."""
        return self._model.get_feature_importance()

    @property
    def predict(self) -> Callable[[np.ndarray], np.ndarray]:
        """Predict using the fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """
        return self._model.predict

    def __init__(self, **params: Any) -> None:
        from catboost import CatBoostRegressor

        self._params = params
        self._model = CatBoostRegressor(**params)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params
            Estimator parameters.
        """
        params = self._model.get_params(deep=deep)

        params.update(self._params)

        return params

    def set_params(self, **params: Any) -> "ModifiedCatBoostRegressor":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self
            Return self.
        """
        for key, value in params.items():
            self._params[key] = value

        self._model.set_params(**params)

        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray, **fit_params: Any
    ) -> "ModifiedCatBoostRegressor":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        if "cat_features" not in fit_params:
            cat_features = get_categorical_cols(X, labels=True)
            fit_params["cat_features"] = cat_features

        self._model.fit(X, y, **fit_params)

        return self


class ModifiedColumnTransformer(BaseEstimator, TransformerMixin):
    """Modified ColumnTransformer.

    Examples
    --------
    >>> from pretools.sklearn.estimators import ModifiedColumnTransformer
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> est = ModifiedColumnTransformer([("features", "passthrough", [0])])
    >>> Xt = est.fit_transform(X)
    """

    def __init__(self, transformers: List[Tuple]) -> None:
        self.transformers = transformers

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ModifiedColumnTransformer":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        self.transformers_ = []

        for name, t, cols in self.transformers:
            if callable(cols):
                cols = cols(X)

            if isinstance(t, BaseEstimator):
                t = clone(t)

                t.fit(X.loc[:, cols], y)

            self.transformers_.append((name, t, cols))

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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        Xs = []

        for _, t, cols in self.transformers_:
            Xt = X.loc[:, cols]

            if isinstance(t, BaseEstimator):
                Xt = t.transform(Xt)
                Xt = pd.DataFrame(Xt)

            if t == "drop":
                continue

            Xs.append(Xt)

        return pd.concat(Xs, axis=1)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit to data, then transform it.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        Xs = []

        self.transformers_ = []

        for name, t, cols in self.transformers:
            if callable(cols):
                cols = cols(X)

            Xt = X.loc[:, cols]

            if isinstance(t, BaseEstimator):
                t = clone(t)

                if hasattr(t, "fit_transform"):
                    Xt = t.fit_transform(Xt, y)
                else:
                    Xt = t.fit(Xt, y).transform(Xt)

                Xt = pd.DataFrame(Xt)

            if t != "drop":
                Xs.append(Xt)

            self.transformers_.append((name, t, cols))

        return pd.concat(Xs, axis=1)


class ModifiedSelectFromModel(BaseEstimator, SelectorMixin):
    """Meta-transformer for selecting features based on importance weights.

    Examples
    --------
    >>> from pretools.sklearn.estimators import ModifiedSelectFromModel
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> est = ModifiedSelectFromModel(LogisticRegression())
    >>> Xt = est.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: bool = True,
        subsample: Union[int, float] = 0.75,
        threshold: Optional[Union[float, str]] = None,
        use_pimp: bool = False,
    ) -> None:
        self.estimator = estimator
        self.random_state = random_state
        self.shuffle = shuffle
        self.subsample = subsample
        self.threshold = threshold
        self.use_pimp = use_pimp

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params: Any
    ) -> "ModifiedSelectFromModel":
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
        if self.subsample < 1.0:
            X, _, y, _ = train_test_split(
                X,
                y,
                random_state=self.random_state,
                train_size=self.subsample,
                shuffle=self.shuffle,
            )

        self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y, **fit_params)

        if self.use_pimp:
            from sklearn.inspection import permutation_importance

            self.feature_importances_ = permutation_importance(
                self.estimator_, X, y, random_state=self.random_state
            ).importances_mean

        else:
            self.feature_importances_ = _get_feature_importances(
                self.estimator_
            )

        return self

    def get_support(self) -> pd.Series:
        """Get a mask of the features selected.

        Returns
        -------
        mask
            Boolean mask.
        """
        threshold = _calculate_threshold(
            self.estimator_, self.feature_importances_, self.threshold
        )

        return self.feature_importances_ >= threshold


class ModifiedStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features.

    Examples
    --------
    >>> from pretools.sklearn.estimators import ModifiedStandardScaler
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> est = ModifiedStandardScaler()
    >>> Xt = est.fit_transform(X)
    """

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ModifiedStandardScaler":
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
        X = check_X(X, estimator=self, force_all_finite="allow-nan")

        self.mean_ = X.mean()
        self.std_ = X.std()
        self.scale_ = self.std_.copy()
        self.scale_[self.scale_ == 0.0] = 1.0

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
        X = check_X(X, estimator=self, force_all_finite="allow-nan")

        return (X - self.mean_) / self.scale_


class ModifiedTargetEncoder(BaseEstimator, TransformerMixin):
    """Modified TargetEncoder.

    Examples
    --------
    >>> from pretools.sklearn.estimators import ModifiedTargetEncoder
    >>> X = [["Cat"], ["Cow"], ["Mouse"], ["Lion"]]
    >>> y = [0, 1, 1, 0]
    >>> est = ModifiedTargetEncoder(cv=2)
    >>> Xt = est.fit_transform(X, y)
    """

    def __init__(
        self,
        cv: Union[BaseCrossValidator, int] = 5,
        dtype: Union[str, Type] = "float64",
        min_samples_leaf: int = 1,
        smoothing: float = 1.0,
    ) -> None:
        self.cv = cv
        self.dtype = dtype
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    def _target_encode_fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Dict[str, pd.Series], float]:
        mapping = {}
        prior = y.mean()

        for col in X:
            grouped = y.groupby(X[col])
            stats = grouped.aggregate(["count", "mean"])
            smoove = sigmoid(
                (stats["count"] - self.min_samples_leaf) / self.smoothing
            )
            smoothing = (1.0 - smoove) * prior + smoove * stats["mean"]
            # smoothing[stats["count"] == 1] = prior
            mapping[col] = smoothing

        return mapping, prior

    def _tartget_encode_transform(
        self, X: pd.DataFrame, mapping: Dict[str, pd.Series], prior: float,
    ) -> pd.DataFrame:
        Xt = X.copy()

        for col in X:
            Xt[col] = Xt[col].map(mapping[col])
            Xt[col] = Xt[col].astype(self.dtype)
            is_null = Xt[col].isnull()
            Xt.loc[is_null, col] = prior

        return Xt

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "ModifiedTargetEncoder":
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
        target_type = type_of_target(y)

        if target_type not in ["binary", "continuous"]:
            raise NotImplementedError(
                "{} is not supported.".format(target_type)
            )

        encoder = LabelEncoder()
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        y = encoder.fit_transform(y)
        y = pd.Series(y, index=X.index)

        self.mapping_, self.prior_ = self._target_encode_fit(X, y)

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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        return self._tartget_encode_transform(X, self.mapping_, self.prior_)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit to data, then transform it.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        Xt
            Transformed data.
        """
        self.fit(X, y, groups=groups)

        target_type = type_of_target(y)
        is_classifier = target_type in ["binary", "multiclass"]
        cv = check_cv(self.cv, y, is_classifier)
        encoder = LabelEncoder()
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        y = encoder.fit_transform(y)
        y = pd.Series(y, index=X.index)
        Xt = np.full_like(X, np.nan, dtype=self.dtype)
        Xt = pd.DataFrame(Xt, columns=X.columns, index=X.index)

        for train, test in cv.split(X, y, groups=groups):
            mapping, prior = self._target_encode_fit(
                X.iloc[train], y.iloc[train]
            )
            Xt.iloc[test] = self._tartget_encode_transform(
                X.iloc[test], mapping, prior
            )

        return Xt


class NAValuesThreshold(BaseEstimator, TransformerMixin):
    """Feature selector that removes features with many missing values.

    Examples
    --------
    >>> from pretools.sklearn.estimators import NAValuesThreshold
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> est = NAValuesThreshold()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "NAValuesThreshold":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        logger = logging.getLogger(__name__)

        cols = self.count_ >= (1.0 - self.threshold) * self.n_samples_
        _, n_features = X.shape
        n_dropped_features = n_features - np.sum(cols)

        logger.info(
            "{} dropped {} features.".format(
                self.__class__.__name__, n_dropped_features
            )
        )

        return X.loc[:, cols]


class NUniqueThreshold(BaseEstimator, TransformerMixin):
    """Feature selector that removes low and high cardinal features.

    Examples
    --------
    >>> from pretools.sklearn.estimators import NUniqueThreshold
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> est = NUniqueThreshold()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(
        self,
        max_freq: Optional[Union[float, int]] = 1.0,
        min_freq: Union[float, int] = 1,
    ) -> None:
        self.max_freq = max_freq
        self.min_freq = min_freq

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "NUniqueThreshold":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

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

        logger.info(
            "{} dropped {} features.".format(
                self.__class__.__name__, n_dropped_features
            )
        )

        return X.loc[:, cols]


class Profiler(BaseEstimator, TransformerMixin):
    """Profiler.

    Examples
    --------
    >>> from pretools.sklearn.estimators import Profiler
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> est = Profiler()
    >>> Xt = est.fit_transform(X, y)
    """

    def __init__(
        self,
        label_col: str = "label",
        max_columns: Optional[int] = None,
        precision: int = 3,
    ) -> None:
        self.label_col = label_col
        self.max_columns = max_columns
        self.precision = precision

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "Profiler":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        data = X.copy()

        if y is not None:
            kwargs = {self.label_col: y}
            data = data.assign(**kwargs)

        logger = logging.getLogger(__name__)
        summary = data.describe(include="all")

        with pd.option_context(
            "display.max_columns",
            self.max_columns,
            "display.precision",
            self.precision,
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
        return check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )


class RandomSeedAveragingClassifier(BaseEstimator, ClassifierMixin):
    """Random seed averaging classifier.

    Examples
    --------
    >>> from pretools.sklearn.estimators import RandomSeedAveragingClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> est = RandomForestClassifier(n_estimators=10)
    >>> est = RandomSeedAveragingClassifier(est)
    >>> X, y = load_iris(return_X_y=True)
    >>> est.fit(X, y)
    RandomSeedAveragingClassifier(...)
    >>> y_pred = est.predict(X)
    """

    @property
    def classes_(self) -> np.ndarray:
        """Class labels."""
        return self.estimators_[0].classes_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances."""
        results = [e.feature_importances_ for e in self.estimators_]

        return np.average(results, axis=0)

    def __init__(
        self,
        estimator: BaseEstimator,
        n_estimators: int = 10,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, **fit_params: Any
    ) -> "RandomSeedAveragingClassifier":
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
        random_state = check_random_state(self.random_state)

        self.estimators_ = []

        for _ in trange(self.n_estimators):
            e = clone(self.estimator)
            seed = random_state.randint(0, MAX_INT)
            to_set = {}
            params = e.get_params(deep=True)

            for key in params:
                if key.endswith("random_state"):
                    to_set[key] = seed

            e.set_params(**to_set)
            e.fit(X, y, **fit_params)

            self.estimators_.append(e)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """
        probas = self.predict_proba(X)
        class_index = np.argmax(probas, axis=1)

        return self.classes_[class_index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        p
            Class probabilities of data.
        """
        results = [e.predict_proba(X) for e in self.estimators_]

        return np.average(results, axis=0)


class RandomSeedAveragingRegressor(BaseEstimator, RegressorMixin):
    """Random seed averaging regressor.

    Examples
    --------
    >>> from pretools.sklearn.estimators import RandomSeedAveragingRegressor
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> est = RandomForestRegressor(n_estimators=10)
    >>> est = RandomSeedAveragingRegressor(est)
    >>> X, y = load_boston(return_X_y=True)
    >>> est.fit(X, y)
    RandomSeedAveragingRegressor(...)
    >>> y_pred = est.predict(X)
    """

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances."""
        results = [e.feature_importances_ for e in self.estimators_]

        return np.average(results, axis=0)

    def __init__(
        self,
        estimator: BaseEstimator,
        n_estimators: int = 10,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, **fit_params: Any
    ) -> "RandomSeedAveragingRegressor":
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
        random_state = check_random_state(self.random_state)

        self.estimators_ = []

        for _ in trange(self.n_estimators):
            e = clone(self.estimator)
            seed = random_state.randint(0, MAX_INT)
            to_set = {}
            params = e.get_params(deep=True)

            for key in params:
                if key.endswith("random_state"):
                    to_set[key] = seed

            e.set_params(**to_set)
            e.fit(X, y, **fit_params)

            self.estimators_.append(e)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """
        results = [e.predict(X) for e in self.estimators_]

        return np.average(results, axis=0)


class RowStatistics(BaseEstimator, TransformerMixin):
    """Row statistics.

    Examples
    --------
    >>> from pretools.sklearn.estimators import RowStatistics
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> est = RowStatistics()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(
        self,
        agg_funcs: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        include_data: bool = False,
        prefix: Optional[str] = None,
    ) -> None:
        self.agg_funcs = agg_funcs
        self.columns = columns
        self.include_data = include_data
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "RowStatistics":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )
        Xt = pd.DataFrame()

        if self.agg_funcs is None:
            agg_funcs = [
                "min",
                "max",
                "mean",
                "std",
                "kurtosis",
                "skew",
            ]
        else:
            agg_funcs = self.agg_funcs

        if self.columns is None:
            columns = X.columns
        else:
            columns = self.columns

        if self.prefix is None:
            prefix = ""
        else:
            prefix = self.prefix

        for agg_func in agg_funcs:
            Xt[f"{prefix}{agg_func}"] = X[columns].agg(agg_func, axis=1)

        if self.include_data:
            Xt = pd.concat([X, Xt], axis=1)

        return Xt


class SortSamples(BaseEstimator, TransformerMixin):
    """Transformer that sorts samples.

    Examples
    --------
    >>> import datetime
    >>> from pretools.sklearn.estimators import SortSamples
    >>> X = [
    ...     [
    ...         datetime.datetime(1999, 1, 1),
    ...         datetime.datetime(1999, 3, 1),
    ...         datetime.datetime(1999, 2, 1),
    ...     ]
    ... ]
    >>> est = SortSamples()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(
        self, copy: bool = True, by: Optional[Union[List[str], str]] = None,
    ) -> None:
        self.copy = copy
        self.by = by

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "SortSamples":
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
        return check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit to data, then transform it.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        Xt
            Transformed data.
        """
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        if self.by is None:
            by = get_time_cols(X, labels=True)
            by = list(by)
        else:
            by = self.by

        if by:
            if self.copy:
                X = X.copy()

            X.sort_values(by, inplace=True)

        return X


class TextStatistics(BaseEstimator, TransformerMixin):
    """Text statistics.

    Examples
    --------
    >>> from pretools.sklearn.estimators import TextStatistics
    >>> X = [["Cat"], ["Cow"], ["Mouse"], ["Lion"]]
    >>> est = TextStatistics()
    >>> Xt = est.fit_transform(X)
    """

    def __init__(
        self, copy: bool = True, dtype: Union[str, Type] = "float64",
    ) -> None:
        self.copy = copy
        self.dtype = dtype

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "TextStatistics":
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
        X = check_X(
            X, dtype=None, estimator=self, force_all_finite="allow-nan"
        )

        if self.copy:
            X = X.copy()

        for col in X:
            X[col] = X[col].str.len()

        X.rename(columns="{}_len".format, inplace=True)

        return X
