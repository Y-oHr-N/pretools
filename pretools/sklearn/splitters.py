"""Splitters."""

from typing import Iterator, Optional, Tuple

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import _num_samples

from ..types import OneDimArrayLikeType, TwoDimArrayLikeType


def _unique_without_sort(array: OneDimArrayLikeType) -> np.ndarray:
    unique, index = np.unique(array, return_index=True)

    return unique[index.argsort()]


class GroupTimeSeriesSplit(_BaseKFold):
    """Time series cross-validator variant with non-overlapping groups.

    Examples
    --------
    >>> import numpy as np
    >>> from pretools.sklearn.splitters import GroupTimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([1, 1, 1, 2, 2, 3])
    >>> cv = GroupTimeSeriesSplit(n_splits=2)
    >>> for train, test in cv.split(X, y, groups):
    ...     X_train, X_test = X[train], X[test]
    ...     y_train, y_test = y[train], y[test]
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        gap: int = 0,
    ) -> None:
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.gap = gap
        self.max_train_size = max_train_size

    def split(
        self,
        X: Optional[TwoDimArrayLikeType] = None,
        y: Optional[OneDimArrayLikeType] = None,
        groups: Optional[OneDimArrayLikeType] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set."""
        unique = _unique_without_sort(groups)

        n_samples = _num_samples(groups)
        n_folds = self.n_splits + 1
        (n_groups,) = unique.shape

        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds ={n_folds} greater than the "
                f"number of groups: {n_groups}."
            )

        test_size = n_groups // n_folds

        if n_groups - self.gap - test_size * self.n_splits <= 0:
            raise ValueError(
                (
                    f"Too many splits={self.n_splits} for number of "
                    f"groups={n_groups} with test_size={test_size} and "
                    f"gap={self.gap}."
                )
            )

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size

            if i + 1 < self.n_splits:
                test_end = (i + 2) * test_size
            else:
                test_end = n_groups

            train_end = test_start - self.gap

            if self.max_train_size is None:
                is_train = np.isin(groups, unique[:train_end])
            else:
                is_train = np.isin(
                    groups,
                    unique[
                        max(0, train_end - self.max_train_size) : train_end
                    ],
                )

            is_test = np.isin(groups, unique[test_start:test_end])

            yield indices[is_train], indices[is_test]
