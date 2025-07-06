# dsl

_WIP Repository for Python implementation of Design-based Supervised Learning_.

Design-based Supervised Learning (DSL) is a framework for unbiased estimation of parameters using imperfect surrogates with arbitrary errors and a subset of randomly sampled gold-standard observations.

This repository provides a Python implementation of DSL (in addition to the [existing R package](https://naokiegami.com/dsl/)). For the initial release, we are targeting the following estimators:

- Mean
- Logistic Regression
- Linear Regression

## Usage

_Note: This is a work in progress. The API may change._

_Also note: This example will not work, as the dataset is too small. This is just a template to show how to use the DSLModel class._

```python
import numpy as np
import pandas as pd
from dsl import DSLModel

# Some dummy data
data = pd.DataFrame({
    "y": [1.0, 2.0, np.nan, np.nan],
    "Q1": [0.9, 2.1, 3.0, 4.2],
    "Q2": [1.1, 1.9, 2.8, 3.9],
    "X1": [10, 20, 30, 40],
    "X2": [5, 15, 25, 35],
    "W1": [100, 200, 300, 400],
    "pi": [0.1, 0.2, 0.3, 0.4],
    "R": [1, 1, 0, 0]
})

# Instantiate DSLModel with dataset
dsl_model = DSLModel(
    # Data args
    data=data,
    gold_standard_col="y",
    predicted_val_cols=["Q1", "Q2"],
    regressor_cols=["X1", "X2"],
    predictor_cols=["W1"],
    sample_weights="pi",
    missing_indicator="R",
    # Estimator args
    model_type="logistic",  # or "logistic", "linear"
    n_cross_fit=2,  # Number of cross-fitting folds
)
dsl_result = dsl_model.fit()
print(dsl_result)
```



## Links to Papers

- [Paper I (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/d862f7f5445255090de13b825b880d59-Paper-Conference.pdf)
- [Paper II (Sociological Methods and Research)](https://journals.sagepub.com/doi/abs/10.1177/00491241251333372?mi=ehikzz)
- [Paper III (Under Review)](https://naokiegami.com/paper/dsl_ss.pdf)

## Companion R Package

See [https://naokiegami.com/dsl/](https://naokiegami.com/dsl/).


