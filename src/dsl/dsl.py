"""
Design-based Supervised Learning (DSL) module.
Authors:
  - Musashi hinck

Notes:

- Initially writing it just for logistic regression.
- TODO: Add linear regression
- TODO: Add fixed effects regression
- Current implementation: input as pandas DataFrame, with strings corresponding to column names.
- Eventual plan: implement like a sklearn estimator. (Does it make sense to have a predict method?)


DSL:

Inputs:
- Data {R_i, R_iY_i, Q_i, X_i, W_i} for i = 1, ..., n
- Known gold-standard probability \pi(Q_i, W_i, X_i) for all i

Algorithm:
1. Randomly partition observation indices into K groups \mathcal{D}_k where k  = 1, ..., K
2. Learn \hat{g}_k from gold-standard documents not in \mathcal{D}_k by predicting Y with (Q, W, X)
3. For documents i \in \mathcal{D}_k, construct bias-corrected pseudo-outcome \tilde{Y}_i^k (eq. 4)
4. Solving the logistic regression moment equation by replacing Y_i with \tilde{Y}_i^k (eq. 5)

Outputs:
- Estimated coefficients \hat{\beta} from logistic regression
- Estimated variance-covariance matrix \hat{V}

Equations:
4. \tilde{Y}_i^k := \hat{g}_k(Q_i, W_i, X_i) + R_i / \pi(Q_i, W_i, X_i) (Y_i - \hat{g}_k(Q_i, W_i, X_i))
5. \sum_{k=1}^K \sum_{i \in \mathcal{D}_k} (\tilde{Y}_i^k - expit(X_i^T \beta))X_i = 0


"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.tree import RandomForestRegressor

from dsl.logit import logistic_regression, ModelResult


class DSLDataset:
    """
    DSLDataset is currently just a wrapper around a pandas DataFrame, in order to have consistent methods for accessing the data.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 y_var: str,
                 x_vars: str|list[str],
                 q_vars: str|list[str],
                 sample_weights: str = None,
                 ) -> None:
        # Checks
        assert y_var in data.columns, "y_var must be a column in the data DataFrame"
        if isinstance(q_vars, str):
            q_vars = [q_vars]
        if isinstance(x_vars, str):
            x_vars = [x_vars]
        assert all(var in data.columns for var in q_vars), "q_vars must be columns in the data DataFrame"
        assert all(var in data.columns for var in x_vars), "x_vars must be columns in the data DataFrame"
        assert sample_weights is None or sample_weights in data.columns, "sample_weights must be a column in the data DataFrame or None"
        # Initialize attributes
        self.dataframe = data
        self.y = self.dataframe[y_var].values()
        self.X = self.dataframe[x_vars].values()
        self.Q = self.dataframe[q_vars].values()
        if sample_weights is None:
            self.w = np.repeat(1 / self.dataframe.shape[0], self.dataframe.shape[0])
        elif isinstance(sample_weights, str):
            self.w = self.dataframe[sample_weights].values() 


class DSLModel(BaseEstimator):
    """
    
    """

    def __init__(
        self,
        model_type: str = "logistic",
        data: pd.DataFrame = None,
        y_var: str = None,
        q_vars: str|list[str] = None,
        x_vars: str|list[str] = None,
        sample_weights: str|list[str] = None,
    ) -> None:
        self._check_inputs(
            model_type=model_type,
            data=data,
            y_var=y_var,
            q_vars=q_vars,
            x_vars=x_vars,
            sample_weights=sample_weights
        )
        self.model_type = model_type
        self.data = data
        self.y_data = data[y_var]
        pass


    def fit(
        self,
        sl_estimator: str="RandomForestRegressor", # This should be an scikit-learn supervised learning estimator
        cross_fit: int = 5,
        sample_split: int = 10,
        random_seed: int = 1234
    ) -> None:
        assert sl_estimator in ["RandomForestRegressor"], "Only RandomForestRegressor is currently supported"
        sl_estimator = RandomForestRegressor(random_state=check_random_state(random_seed))


    def _check_inputs(self, 
        model_type: str = "logistic",
        data: pd.DataFrame = None,
        y_var: str = None,
        q_vars: str|list[str] = None,
        x_vars: str|list[str] = None,
        sample_weights: str|list[str] = None,
    ) -> None:
        # Check inputs
        assert model_type in ["logistic"], "Model type not supported" # TODO: Add linear regression, fixed effects regression
