r"""
Design-based Supervised Learning (DSL) module.
Authors:
  - Musashi hinck

Notes:

- Initially writing it just for logistic regression.
- TODO: Add linear regression
- TODO: Add fixed effects regression
- Idea: implement like a sklearn estimator. (Does it make sense to have a predict method?)
- NOTE: how to manage multiprocessing? We can run over sample splits + cross validation folds in parallel, but we need to allocate threads appropriately.


DSL Algorithm
=============

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
from sklearn.model_selection import KFold

from dsl.logit import logistic_regression, ModelResult


class DSLDataset:
    """
    DSLDataset is currently just a wrapper around a pandas DataFrame, in order to have consistent methods for accessing the data.

    TODO:
    - Add methods for partitioning the data into sample splits for DSL (note--should be stratified on presence of gold standard)
    - Add methods for partitioning the data into K folds for cross-validation
    - Add methods for getting the data in a format suitable for scikit-learn estimators
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y_var: str,
        x_vars: str | list[str],
        q_vars: str | list[str],
        sample_weights: str = None,
    ) -> None:
        # Checks
        assert y_var in data.columns, "y_var must be a column in the data DataFrame"
        if isinstance(q_vars, str):
            q_vars = [q_vars]
        if isinstance(x_vars, str):
            x_vars = [x_vars]
        assert all(
            var in data.columns for var in q_vars
        ), "q_vars must be columns in the data DataFrame"
        assert all(
            var in data.columns for var in x_vars
        ), "x_vars must be columns in the data DataFrame"
        assert (
            sample_weights is None or sample_weights in data.columns
        ), "sample_weights must be a column in the data DataFrame or None"
        # Initialize attributes
        self.dataframe = data
        self.y = self.dataframe[y_var].values()
        self.X = self.dataframe[x_vars].values()
        self.Q = self.dataframe[q_vars].values()
        if sample_weights is None:
            self.w = np.repeat(1 / self.dataframe.shape[0], self.dataframe.shape[0])
        elif isinstance(sample_weights, str):
            self.w = self.dataframe[sample_weights].values()

    def sample_split(self, n_splits: int) -> None:
        """
        Split the dataset into `n_splits` partitions.
        This method should be implemented to create sample splits for DSL.
        """
        pass
    
    def get_ss_partition(self, ss_idx: int) -> "DSLDataset":
        """
        Get the sample split partition with index `ss_idx`.
        This method should be implemented to return a DSLDataset for the specified sample split.
        """
        # Placeholder implementation, should be replaced with actual logic
        return self

    def get_train_test(self, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the training and testing data for the specified indices.
        """
        y_train = self.y[train_idx]
        X_train = self.X[train_idx]
        X_test = self.X[test_idx]
        return y_train, X_train, X_test

class DSLModel(BaseEstimator):
    """
    """

    def __init__(
        self,
        model_type: str = "logistic",
        data: pd.DataFrame = None,
        y_var: str = None,
        q_vars: str | list[str] = None,
        x_vars: str | list[str] = None,
        sample_weights: str | list[str] = None,
        random_seed: int=634,
    ) -> None:
        self._validate_inputs(
            model_type=model_type,
            data=data,
            y_var=y_var,
            q_vars=q_vars,
            x_vars=x_vars,
            sample_weights=sample_weights,
        )
        self.model_type = model_type
        self.data = data
        self.random_seed = random_seed
        pass

    def fit(
        self,
        sl_estimator: str = "RandomForestRegressor",  # This should be an scikit-learn supervised learning estimator
        n_sample_split: int = 10,
        n_cross_fit: int = 5,
        sl_kwargs: dict = {},
    ) -> None:
        """
        Split into `sample_split` partitions S
        For each partition s in S
            Split into K folds
            For each fold k in K
                Fit model g_k on data not in fold k
                Predict ytilde in k
            Concatenate ytilde_k to create ytilde_s
            Fit logistic regression ytilde_s ~ X_s
            Record params beta_s
        Average (median?) beta_s across splits
        """
        self.data.sample_split(n_sample_split)
        param_estimates = []
        for ss_idx in range(n_sample_split):
            param_estimates.append(
                self._fit_split(ss_idx, n_cross_fit, sl_estimator, sl_kwargs)
            )

    def _fit_split(self, ss_idx: int, n_cross_fit, sl_estimator, sl_kwargs):
        data_ss = self.data.get_ss_partition(ss_idx)
        kfold_iterator = KFold(n_splits=n_cross_fit, random_seed=self.random_seed)
        ghat_list = [
            sl_estimator(**sl_kwargs|{'random_seed': self.random_seed + (ss_idx*n_cross_fit) + k_idx})
            for k_idx in n_cross_fit
        ]

        for k_idx, (train_idx, test_idx) in kfold_iterator(data_ss):
            y_train, X_train, X_test = data_ss.get_train_test(train_idx, test_idx)
            ghat_list[k_idx].fit(y_train, X_train)
            ghat_list[k_idx].predict(X_test)



    def _validate_inputs(
        self,
        model_type: str = "logistic",
        data: pd.DataFrame = None,
        y_var: str = None,
        q_vars: str | list[str] = None,
        x_vars: str | list[str] = None,
        sample_weights: str | list[str] = None,
    ) -> None:
        # Check inputs
        assert model_type in [
            "logistic"
        ], "Model type not supported"  # TODO: Add linear regression, fixed effects regression
