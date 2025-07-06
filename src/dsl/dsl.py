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

from enum import Enum
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from dsl.estimators import LogisticRegression, MeanEstimator, ModelResult
from dsl.data_utils import DSLDataset


class DSLModelType(Enum):
    LOGISTIC = "logistic"
    LINEAR = "linear"
    MEAN = "mean"

    @staticmethod
    def get_estimator(model_type: str):
        match model_type:
            case DSLModelType.LOGISTIC.value:
                return LogisticRegression().fit
            case DSLModelType.LINEAR.value:
                return LinearRegression
            case DSLModelType.MEAN.value:
                return MeanEstimator().fit
            case _:
                raise ValueError(f"Unknown model_type: {model_type}")


class DSLModel:

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
        random_state: int = 634,
        n_cross_fit: int = 5,
        sl_estimator: BaseEstimator = RandomForestRegressor,
        cv_iterator: BaseCrossValidator = StratifiedKFold,
        sl_kwargs: dict = {},
        cv_iterator_kwargs: dict = {},
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
        self.random_state = random_state
        # Initialize cross-fitting parameters
        self._sl_estimators = [
            sl_estimator(**sl_kwargs | {"random_state": self.random_state + k_idx})
            for k_idx in range(n_cross_fit)
        ]
        self.cv = cv_iterator(**cv_iterator_kwargs | {"n_splits": n_cross_fit})
        self._splits = list(self.cv.split(range(len(self.data)), self.data.R))

    def fit(
        self,
    ) -> ModelResult:
        """
        Split into K folds
        For each fold k in K
            Fit model g_k on data not in fold k
            Predict ytilde in k
        Concatenate ytilde_k to create ytilde
        Fit DSL estimator on ytilde ~ X
        """
        # Init SL estimators
        self.pseudo_outcome = np.zeros_like(self.data.y, dtype=float)
        for fold_idx, (train_idx, test_idx) in enumerate(self._splits):
            self.pseudo_outcome[test_idx] = self._fit_fold(fold_idx, train_idx, test_idx)
        # Fit DSL estimator on outcome
        estimator_args = {"Y": self.pseudo_outcome}
        if self.data.X is not None:
            estimator_args |= {"X": self.data.X}
        self.result: ModelResult = self.dsl_estimator(**estimator_args)
        return self.result

    def _fit_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> np.ndarray:
        r"""
        Fits \hat{g}_k(Q_i, W_i, X_i) on training data and predicts ytilde on test data
        """
        # Get train and test indices for the given fold
        train_data = self.data[train_idx]
        test_data = self.data[test_idx]
        # Fit ghat
        self._sl_estimators[fold_idx].fit(X=train_data.QWX, y=train_data.y)
        # Predict ytilde on test data
        ghat = self._sl_estimators[fold_idx].predict(test_data.QWX)
        ytilde = ghat + (test_data.R / test_data.pi) * (test_data.y - ghat)
        return ytilde
