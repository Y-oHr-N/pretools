# pretools

![Python package](https://github.com/Y-oHr-N/pretools/workflows/Python%20package/badge.svg?branch=master)
[![PyPI](https://img.shields.io/pypi/v/pretools)](https://pypi.org/project/pretools/)
[![PyPI - License](https://img.shields.io/pypi/l/pretools)](https://pypi.org/project/pretools/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Y-oHr-N/pretools/master)

Preparation tools for machine learning.

## Examples

```python
from pretools.estimators import *
from sklearn.datasets import load_boston
from sklearn.pipeline import make_pipeline

X, y = load_boston(return_X_y=True)
model = ModifiedCatBoostRegressor(random_state=0, verbose=100)
model = make_pipeline(
    Profiler(),
    Astype(),
    NUniqueThreshold(max_freq=None),
    DropCollinearFeatures(method="spearman", random_state=0),
    ClippedFeatures(),
    ModifiedStandardScaler(),
    ModifiedSelectFromModel(model, random_state=0, threshold=1e-06),
    CombinedFeatures(),
    ModifiedSelectFromModel(model, random_state=0, threshold=1e-06),
    model,
)

model.fit(X, y)
```

## Installation

```
pip install pretools
```

## Testing

```
python setup.py test
```
