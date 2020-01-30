"""Estimators."""

import itertools
import logging

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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

try:  # scikit-learn<=0.21
    from sklearn.feature_selection.from_model import _calculate_threshold
    from sklearn.feature_selection.from_model import _get_feature_importances
except ImportError:
    from sklearn.feature_selection._from_model import _calculate_threshold
    from sklearn.feature_selection._from_model import _get_feature_importances

from .utils import get_categorical_cols
from .utils import get_numerical_cols
from .utils import get_time_cols
from .utils import get_unknown_cols
from .utils import sigmoid


class Astype(BaseEstimator, TransformerMixin):
    """Astype."""

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
        X = pd.DataFrame(X)
        Xt = X.copy()
        numerical_cols = get_numerical_cols(X, labels=True)
        unknown_cols = get_unknown_cols(X, labels=True)

        if len(numerical_cols) > 0:
            Xt[numerical_cols] = Xt[numerical_cols].astype("float32")

        if len(unknown_cols) > 0:
            Xt[unknown_cols] = Xt[unknown_cols].astype("category")

        return Xt


class CalendarFeatures(BaseEstimator, TransformerMixin):
    """Calendar features."""

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
        X = pd.DataFrame(X)
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
    """Clipped features."""

    def __init__(self, high: float = 0.99, low: float = 0.01) -> None:
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
        X = pd.DataFrame(X)

        self.n_samples_, self.n_features_ = X.shape

        if self.max_features is None:
            self.max_features_ = np.inf
        elif self.max_features == "auto":
            self.max_features_ = self.n_samples_ - self.n_features_ - 1
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
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()
        is_numerical = get_numerical_cols(X)
        numerical_cols = X.columns[is_numerical]
        other_cols = X.columns[~is_numerical]

        n_features = 0

        logger = logging.getLogger(__name__)

        for col1, col2 in itertools.combinations(other_cols, 2):
            for operand in self._operands:
                if n_features >= self.max_features_:
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

        for col1, col2 in itertools.combinations(numerical_cols, 2):
            for operand in self._operands:
                if n_features >= self.max_features_:
                    break

                func = getattr(np, operand)

                Xt["{}_{}_{}".format(operand, col1, col2)] = func(
                    X[col2], X[col2]
                )

                n_features += 1

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
    """Diff features."""

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
        X = pd.DataFrame(X)
        Xt = X.diff()

        Xt.rename(columns="{}_diff".format, inplace=True)

        if self.include_data:
            Xt = pd.concat([X, Xt], axis=1)

        return Xt


class DropCollinearFeatures(BaseEstimator, TransformerMixin):
    """Feature selector that removes collinear features."""

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
        X = pd.DataFrame(X)

        X, _, = train_test_split(
            X,
            random_state=self.random_state,
            train_size=self.subsample,
            shuffle=self.shuffle,
        )

        self.corr_ = X.corr(method=self.method)

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

        logger.info(
            "{} dropped {} features.".format(
                self.__class__.__name__, n_dropped_features
            )
        )

        return X.loc[:, cols]


class ModifiedCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """Modified CatBoostClassifier."""

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
        X = pd.DataFrame(X)
        y = self._encoder.fit_transform(y)
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
    """Modified CatBoostRegressor."""

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
        X = pd.DataFrame(X)
        cat_features = get_categorical_cols(X, labels=True)
        fit_params["cat_features"] = cat_features

        self._model.fit(X, y, **fit_params)

        return self


class ModifiedColumnTransformer(BaseEstimator, TransformerMixin):
    """Modified ColumnTransformer."""

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
        X = pd.DataFrame(X)

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
        X = pd.DataFrame(X)
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
        X = pd.DataFrame(X)
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


class ModifiedSelectFromModel(BaseEstimator, TransformerMixin):
    """Meta-transformer for selecting features based on importance weights."""

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
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=self.random_state,
            train_size=self.subsample,
            shuffle=self.shuffle,
        )

        self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X_train, y_train, **fit_params)

        if self.use_pimp:
            from sklearn.inspection import permutation_importance

            self.feature_importances_ = permutation_importance(
                self.estimator_, X_test, y_test, random_state=self.random_state
            ).importances_mean

        else:
            self.feature_importances_ = _get_feature_importances(
                self.estimator_
            )

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

        threshold = _calculate_threshold(
            self.estimator_, self.feature_importances_, self.threshold
        )

        logger = logging.getLogger(__name__)

        cols = self.feature_importances_ >= threshold
        _, n_features = X.shape
        n_dropped_features = n_features - np.sum(cols)

        logger.info(
            "{} dropped {} features.".format(
                self.__class__.__name__, n_dropped_features
            )
        )

        return X.loc[:, cols]


class ModifiedStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features."""

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
        X = pd.DataFrame(X)

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
        X = pd.DataFrame(X)

        return (X - self.mean_) / self.scale_


class ModifiedTargetEncoder(BaseEstimator, TransformerMixin):
    """Modified TargetEncoder."""

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
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, pd.Series]:
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

        return mapping

    def _tartget_encode_transform(
        self,
        X: pd.DataFrame,
        mapping: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        Xt = X.copy()

        for col in X:
            Xt[col] = Xt[col].map(mapping[col])

        return Xt.astype(self.dtype)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None
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

        # if target_type != "continuous":
        #     raise NotImplementedError()

        X = pd.DataFrame(X)
        y = pd.Series(y, index=X.index)

        self.mapping_ = self._target_encode_fit(X, y)

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

        return self._tartget_encode_transform(X, self.mapping_)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None
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
        is_classifier = target_type in [
            "binary",
            "multiclass",
            "multiclass-output",
            "multilabel-indicator",
        ]
        cv = check_cv(self.cv, y, is_classifier)
        X = pd.DataFrame(X)
        y = pd.Series(y, index=X.index)
        Xt = np.full_like(X, np.nan, dtype=self.dtype)
        Xt = pd.DataFrame(Xt, columns=X.columns, index=X.index)

        for train, test in cv.split(X, y, groups=groups):
            mapping = self._target_encode_fit(X.iloc[train], y.iloc[train])
            Xt.iloc[test] = self._tartget_encode_transform(
                X.iloc[test], mapping
            )

        return Xt


class NAValuesThreshold(BaseEstimator, TransformerMixin):
    """Feature selector that removes features with many missing values."""

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

        logger.info(
            "{} dropped {} features.".format(
                self.__class__.__name__, n_dropped_features
            )
        )

        return X.loc[:, cols]


class NUniqueThreshold(BaseEstimator, TransformerMixin):
    """Feature selector that removes low and high cardinal features."""

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

        logger.info(
            "{} dropped {} features.".format(
                self.__class__.__name__, n_dropped_features
            )
        )

        return X.loc[:, cols]


class Profiler(BaseEstimator, TransformerMixin):
    """Profiler."""

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
        data = pd.DataFrame(X)

        if y is not None:
            kwargs = {self.label_col: y}
            data = X.assign(**kwargs)

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
        return pd.DataFrame(X)


class RowStatistics(BaseEstimator, TransformerMixin):
    """Row statistics."""

    def __init__(self, dtype: Union[str, Type] = "float64") -> None:
        self.dtype = dtype

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
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        is_null = X.isnull()

        Xt["number_of_na_values"] = is_null.sum(axis=1)

        return Xt.astype(self.dtype)


class SortSamples(BaseEstimator, TransformerMixin):
    """Transformer that sorts samples."""

    def __init__(self, by: Optional[Union[List[str], str]] = None) -> None:
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
        return pd.DataFrame(X)

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
        X = pd.DataFrame(X)

        if self.by is None:
            by = get_time_cols(X, labels=True)
            by = list(by)
        else:
            by = self.by

        if by:
            X = X.sort_values(by)

        return X


class TextStatistics(BaseEstimator, TransformerMixin):
    """Text statistics."""

    def __init__(self, dtype: Union[str, Type] = "float64") -> None:
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
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        for col in X:
            Xt["{}_len".format(col)] = X[col].str.len()

        return Xt
