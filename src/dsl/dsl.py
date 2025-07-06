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
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from dsl.logit import logistic_regression, mean_estimator, ModelResult
from enum import Enum
from sklearn.linear_model import LinearRegression


class DSLModelType(Enum):
    LOGISTIC = "logistic"
    LINEAR = "linear"
    MEAN = "mean"

    @staticmethod
    def get_estimator(model_type: str):
        match model_type:
            case DSLModelType.LOGISTIC.value:
                return logistic_regression
            case DSLModelType.LINEAR.value:
                return LinearRegression
            case DSLModelType.MEAN.value:
                return mean_estimator
            case _:
                raise ValueError(f"Unknown model_type: {model_type}")


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
        assert all(
            isinstance(col, str) for col in self.predicted_val_cols
        ), "predicted_val_cols must be a string or list of strings"
        assert all(
            isinstance(col, str) for col in self.regressor_cols
        ), "regressor_cols must be a string or list of strings"

    def check_columns(self, data: pd.DataFrame) -> None:
        """
        Check that the columns specified in the dataclass are present
        """
        assert (
            self.gold_standard_col in data.columns
        ), f"gold_standard_col '{self.gold_standard_col}' not found in DataFrame"
        for col in self.predicted_val_cols:
            assert (
                col in data.columns
            ), f"predicted_val_col '{col}' not found in DataFrame"
        for col in self.regressor_cols:
            assert col in data.columns, f"regressor_col '{col}' not found in DataFrame"
        if self.predictor_cols is not None:
            for col in self.predictor_cols:
                assert (
                    col in data.columns
                ), f"predictor_col '{col}' not found in DataFrame"
        if self.sample_weights is not None:
            assert (
                self.sample_weights in data.columns
            ), f"sample_weights '{self.sample_weights}' not found in DataFrame"
        if self.missing_indicator is not None:
            assert (
                self.missing_indicator in data.columns
            ), f"missing_indicator '{self.missing_indicator}' not found in DataFrame"


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
        gold_standard_col: str,  # Y
        predicted_val_cols: str | list[str],  # Q
        regressor_cols: str | list[str],  # X
        predictor_cols: str | list[str] = None,  # W
        sample_weights: str = None,  # pi
        missing_indicator: str = None,  # R, if None then use notna
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
        self.Q = self.dataframe[self._col_ref.predicted_val_cols].values
        self.X = (
            self.dataframe[self._col_ref.regressor_cols].values
            if self.col_ref.regressor_cols
            else None
        )
        self.W = (
            self.dataframe[self._col_ref.predictor_cols].values
            if self._col_ref.predictor_cols
            else None
        )
        if sample_weights is None:
            self.pi = np.repeat(1 / self.dataframe.shape[0], self.dataframe.shape[0])
        elif isinstance(sample_weights, str):
            self.pi = self.dataframe[sample_weights].values
        if missing_indicator is None:
            self.R = self.dataframe[gold_standard_col].notna().values
        elif isinstance(missing_indicator, str):
            self.R = self.dataframe[missing_indicator].values
        self.random_seed = random_seed
        # Convenience accessors
        self.QWX = np.hstack(
            self.Q,
            self.W if self.W is not None else None,
            self.X if self.X is not None else None,
        )

    def __getitem__(
        self, idxs: int | list | slice | np.ndarray | pd.Series
    ) -> "DSLDataset":
        """
        Return a new DSLDataset containing only the rows at the given indices.
        Accepts any valid pandas indexer (int, list, slice, boolean array, etc).
        """
        new_data = self.dataframe.iloc[idxs].copy()
        return DSLDataset(
            data=new_data,
            gold_standard_col=self._col_ref.gold_standard_col,
            predicted_val_cols=self._col_ref.predicted_val_cols,
            regressor_cols=self._col_ref.regressor_cols,
            predictor_cols=self._col_ref.predictor_cols,
            sample_weights=self._col_ref.sample_weights,
            missing_indicator=self._col_ref.missing_indicator,
            random_seed=self.random_seed,
        )


class DSLModel(BaseEstimator):

    def __init__(
        self,
        data: pd.DataFrame,
        gold_standard_col: str,
        predicted_val_cols: str | list[str],
        regressor_cols: str | list[str] | None = None,
        predictor_cols: str | list[str] | None = None,
        sample_weights: str | None = None,
        missing_indicator: str | None = None,
        model_type: str = "logistic",
        random_seed: int = 634,
    ) -> None:
        self.data = DSLDataset(
            data=data,
            gold_standard_col=gold_standard_col,
            predicted_val_cols=predicted_val_cols,
            regressor_cols=regressor_cols,
            predictor_cols=predictor_cols,
            sample_weights=sample_weights,
            missing_indicator=missing_indicator,
        )
        self.model_type = model_type
        self.dsl_estimator = DSLModelType.get_estimator(model_type)
        self.random_seed = random_seed

    def fit(
        self,
        n_cross_fit: int = 5,
        sl_estimator: BaseEstimator = RandomForestRegressor,
        cv_iterator: BaseCrossValidator = StratifiedKFold,
        sl_kwargs: dict = {},
        cv_iterator_kwargs: dict = {},
    ) -> None:
        """
        Split into K folds
        For each fold k in K
            Fit model g_k on data not in fold k
            Predict ytilde in k
        Concatenate ytilde_k to create ytilde
        Fit DSL estimator on ytilde ~ X
        """
        # Init SL estimators
        self._sl_estimators = [
            sl_estimator(**sl_kwargs | {"random_seed": self.random_seed + k_idx})
            for k_idx in n_cross_fit
        ]
        self.cv = cv_iterator(**cv_iterator_kwargs | {"n_splits": n_cross_fit})
        self._splits = list(self.cv.split(self.data, self.data[self.y_var]))
        self.pseudo_outcome = np.zeros_like(self.data.y, dtype=float)
        for fold_idx, (train_idx, test_idx) in enumerate(self._splits):
            self.pseudo_outcome[test_idx] = self._fit_fold(fold_idx)
        # Fit DSL estimator on outcome
        estimator_args = {"Y": self.pseudo_outcome} | (
            {"X": self.data.X} if self.data.X else None
        )
        self.result: ModelResult = self.dsl_estimator(**estimator_args)

    def _fit_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> np.ndarray:
        "Fits \hat{g}_k(Q_i, W_i, X_i) on training data and predicts ytilde on test data"
        # Get train and test indices for the given fold
        train_data = self.data[train_idx]
        test_data = self.data[test_idx]
        # Fit ghat
        self._sl_estimators[fold_idx].fit(X=train_data.QWX, y=train_data.y)
        # Predict ytilde on test data
        ghat = self._sl_estimators[fold_idx].predict(test_data.QWX)
        ytilde = ghat + (test_data.R / test_data.pi) * (test_data.y - ghat)
        return ytilde
