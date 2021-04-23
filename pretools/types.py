"""Types."""

import pathlib
from typing import Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

PathLikeType = Union[pathlib.Path, str]
OneDimArrayLikeType = Union[jnp.ndarray, np.ndarray, pd.Series]
TwoDimArrayLikeType = Union[jnp.ndarray, np.ndarray, pd.DataFrame, spmatrix]
