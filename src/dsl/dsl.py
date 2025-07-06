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

NOTE: Missing the bias correction term in the current implementation.

"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

from dsl.logit import logistic_regression, ModelResult


@dataclass
class DSLDatasetColumns:
    """
    DSLDatasetColumns is a dataclass that holds the column names for the DSLDataset.
    It is used to ensure consistent access to the columns in the dataset.
    """
    gold_standard_col: str = field(default=None)
    predicted_val_cols: str | list[str] = field(default=None)
    regressor_cols: str | list[str] = field(default=None)
    predictor_cols: str | list[str] = field(default=None)
    sample_weights: str = field(default=None)
    missing_indicator: str = field(default=None)
    
    def __post_init__(self):
        # Ensure that predicted_val_cols and regressor_cols are lists
        if isinstance(self.predicted_val_cols, str):
            self.predicted_val_cols = [self.predicted_val_cols]
        if isinstance(self.regressor_cols, str):
            self.regressor_cols = [self.regressor_cols]
        if self.predictor_cols is not None:
            if isinstance(self.predictor_cols, str):
                self.predictor_cols = [self.predictor_cols]
        # Validate that all columns are strings
        assert all(isinstance(col, str) for col in self.predicted_val_cols), "predicted_val_cols must be a string or list of strings"
        assert all(isinstance(col, str) for col in self.regressor_cols), "regressor_cols must be a string or list of strings"
    
    def check_columns(self, data: pd.DataFrame) -> None:
        """
        Check that the columns specified in the dataclass are present
        """
        assert self.gold_standard_col in data.columns, f"gold_standard_col '{self.gold_standard_col}' not found in DataFrame"
        for col in self.predicted_val_cols:
            assert col in data.columns, f"predicted_val_col '{col}' not found in DataFrame"
        for col in self.regressor_cols:
            assert col in data.columns, f"regressor_col '{col}' not found in DataFrame"
        if self.predictor_cols is not None:
            for col in self.predictor_cols:
                assert col in data.columns, f"predictor_col '{col}' not found in DataFrame"
        if self.sample_weights is not None:
            assert self.sample_weights in data.columns, f"sample_weights '{self.sample_weights}' not found in DataFrame"
        if self.missing_indicator is not None:
            assert self.missing_indicator in data.columns, f"missing_indicator '{self.missing_indicator}' not found in DataFrame"


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
        gold_standard_col: str, # Y
        predicted_val_cols: str | list[str], # Q
        regressor_cols: str | list[str], # X
        predictor_cols: str | list[str] = None, # W
        sample_weights: str = None, # pi
        missing_indicator: str = None, # R, if None then use notna
        random_seed: int = 634,  # Random seed for reproducibility
    ) -> None:
        # Init and validate columns
        self._col_ref = DSLDatasetColumns(
            gold_standard_col=gold_standard_col,
            predicted_val_cols=predicted_val_cols,
            regressor_cols=regressor_cols,
            predictor_cols=predictor_cols,
            sample_weights=sample_weights,
            missing_indicator=missing_indicator,
        )
        self._col_ref.check_columns(data)
        # Initialize attributes
        self.dataframe = data
        self.y = self.dataframe[self._col_ref.gold_standard_col].values
        self.X = self.dataframe[self._col_ref.regressor_cols].values
        self.Q = self.dataframe[self._col_ref.predicted_val_cols].values
        self.W = self.dataframe[self._col_ref.predictor_cols].values if self._col_ref.predictor_cols else None
        if sample_weights is None:
            self.pi = np.repeat(1 / self.dataframe.shape[0], self.dataframe.shape[0])
        elif isinstance(sample_weights, str):
            self.pi = self.dataframe[sample_weights].values
        if missing_indicator is None:
            self.R = self.dataframe[gold_standard_col].notna().values
        elif isinstance(missing_indicator, str):
            self.R = self.dataframe[missing_indicator].values
        self.random_seed = random_seed

    def generate_sample_splits(self, n_splits: int) -> None:
        """
        Split the dataset into `n_splits` partitions.
        This method should be implemented to create sample splits for DSL.
        """
        ss_split_iterator = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_seed
        )
        self.sample_splits = [(train_idx, test_idx) for train_idx, test_idx in ss_split_iterator.split(self.X, self.y)]
        
    
    def get_ss_partition(self, ss_idx: int) -> "DSLDataset":
        """
        Get the sample split partition with index `ss_idx`.
        This method should be implemented to return a DSLDataset for the specified sample split.
        """
        # Placeholder implementation, should be replaced with actual logic
        if not hasattr(self, 'sample_splits'):
            raise ValueError("Sample splits have not been created. Call `generate_sample_splits` first.")
        split_idx, _ = self.sample_splits[ss_idx]
        return DSLDataset(
            data=self.dataframe.iloc[split_idx],
            gold_standard_col=self._col_ref.gold_standard_col,
            predicted_val_cols=self._col_ref.predicted_val_cols,
            regressor_cols=self._col_ref.regressor_cols,
            predictor_cols=self._col_ref.predictor_cols,
            sample_weights=self._col_ref.sample_weights,
            missing_indicator=self._col_ref.missing_indicator,
            random_seed=self.random_seed,
        )

    def get_train_test(
        self, train_idx: np.ndarray, test_idx: np.ndarray
    ) -> tuple[]:
        """
        Get the training and testing data for the specified indices.
        """
        y_train = self.y[train_idx]
        X_train = self.X[train_idx]
        Q_train = self.Q[train_idx]
        W_train = self.W[train_idx] if self.W is not None else None

        X_test = self.X[test_idx]
        Q_test = self.Q[test_idx]
        W_test = self.W[test_idx] if self.W is not None else None
        return y_train, X_train, Q_train, W_train, X_test, Q_test, W_test
    

class DSLModel(BaseEstimator):

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
        # Average the parameter estimates across sample splits

    def _fit_split(self, ss_idx: int, n_cross_fit, sl_estimator, sl_kwargs) -> ModelResult:
        data_ss = self.data.get_ss_partition(ss_idx)
        kfold_iterator = StratifiedKFold(n_splits=n_cross_fit, random_seed=self.random_seed)
        ghat_list = [
            sl_estimator(**sl_kwargs|{'random_seed': self.random_seed + (ss_idx*n_cross_fit) + k_idx})
            for k_idx in n_cross_fit
        ]
        ytilde_list = []

        for k_idx, (train_idx, test_idx) in kfold_iterator(data_ss):
            y_train, X_train, Q_train, W_train, X_test, Q_test, W_test = data_ss.get_train_test(train_idx, test_idx)
            ytilde = self._fit_fold(y_train, X_train, Q_train, W_train, X_test, Q_test, W_test, ghat_list[k_idx])
            ytilde_list.append(ytilde)
        # Concatenate ytilde_k to create ytilde_s
        ytilde_s = np.concatenate(ytilde_list, axis=0)
        # Fit logistic regression ytilde_s ~ X_s
        model_result = logistic_regression(
            ytilde_s,
            data_ss.X,
            sample_weights=data_ss.pi,
            random_seed=self.random_seed + ss_idx
        )
        return model_result


    def _fit_fold(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        Q_train: np.ndarray,
        W_train: np.ndarray,
        X_test: np.ndarray,
        Q_test: np.ndarray,
        W_test: np.ndarray,
        ghat: BaseEstimator,
    ) -> np.ndarray:
        "Fits \hat{g}_k(Q_i, W_i, X_i) on training data and predicts ytilde on test data"
        qwx_train = np.hstack((Q_train, W_train, X_train)) if W_train is not None else np.hstack((Q_train, X_train))
        ghat.fit(qwx_train, y_train)
        # Predict ytilde on test data
        qwx_test = np.hstack((Q_test, W_test, X_test)) if W_test is not None else np.hstack((Q_test, X_test))
        ytilde = ghat.predict(qwx_test)
        return ytilde


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
