"""
Implementations of estimators for Design-based Statistical Learning
Authors:
  - Musashi Hinck

Currently implemented:
- Logistic Regression (Method of Moments)
- Mean Estimator

"""


import numpy as np
from typing import Optional
from numpy.typing import NDArray
from scipy.optimize import minimize
from dataclasses import dataclass

from .helpers import (
    _scale,
    _logistic,
    _confidence_intervals,
    _logit_objective,
    _logit_jacobian,
    _mean_estimator
)


@dataclass
class ModelResult:
    """
    A dataclass to hold the results of a model estimation.
    """
    coef: NDArray[np.float64]
    se: NDArray[np.float64]
    vcov: NDArray[np.float64]|None

    def confidence_intervals(self, alpha: float = 0.05) -> NDArray[np.float64]:
        """
        Calculate confidence intervals for the coefficients.

        Args:
            alpha (float): Significance level for the confidence interval (default: 0.05)

        Returns:
            NDArray[np.float64]: Confidence intervals for the coefficients
        """
        return _confidence_intervals(self.coef, self.se, alpha)


class LogisticRegression:
    def __init__(self, lam: Optional[float] = 1e-5, minimizer_options: dict={"maxiter": 1000}):
        """
        Initialize the LogisticRegression estimator.

        Args:
            lam (Optional[float]): Regularization parameter (default: 1e-5)
        """
        self.lam = lam
        self.minimizer_options = minimizer_options
    
    def fit(self, Y: NDArray[np.float64], X: NDArray[np.float64]) -> ModelResult:
        """
        Fit the logistic regression model to the data.

        Args:
            Y (NDArray[np.float64]): Dependent variable
            X (NDArray[np.float64]): Independent variables (including a column of ones for intercept)

        Returns:
            ModelResult: The result of the model estimation

        """
        # Pre-compute inputs to minimizer
        num_obs: int = X.shape[0]
        num_covariates: int = X.shape[1]
        x0: NDArray = np.zeros((num_covariates,))
        X_scaled: NDArray = np.concatenate(
            (np.ones((num_obs, 1)), _scale(X[:, 1:])), axis=1
        )
        cons: int = 2 / (num_covariates * num_obs**2)
        X_scaled_squared: NDArray = X_scaled**2

        # Reformat objective and jacobian for minimize
        def _obj(x, *args):
            Y, X_scaled, _, _, self.lam = [*args]
            beta = x
            return _logit_objective(Y, X_scaled, beta, self.lam)

        def _jac(x, *args):
            Y, X, X2, cons, self.lam = [*args]
            beta = x
            return _logit_jacobian(Y, X, X2, beta, cons, self.lam)

        # Optimize
        optimizer_output = minimize(
            fun=_obj,
            x0=x0,
            args=(Y, X_scaled, X_scaled_squared, cons, self.lam),
            jac=_jac,
            method="L-BFGS-B",
            options=self.minimizer_options
        )

        # Collect
        beta_hat: NDArray[np.float64] = optimizer_output.x

        # Standard Error
        pi = _logistic(X_scaled @ beta_hat)
        res = Y - pi
        phi = X_scaled * res[:, np.newaxis]
        omega = (phi.T @ phi) / num_obs
        X_1 = X_scaled * pi[:, np.newaxis]
        X_0 = X_scaled * (1 - pi)[:, np.newaxis]
        M = (X_1.T @ X_0) / num_obs
        M_inv = np.linalg.pinv(M)
        V0 = (M_inv @ omega @ M_inv) / num_obs

        # Recalibrate
        mean_X = np.mean(X[:, 1:], axis=0)
        sd_X = np.std(X[:, 1:], axis=0, ddof=1)
        D_1 = np.concatenate((np.array([1]), -mean_X / sd_X))
        D_2 = np.concatenate((np.zeros((len(sd_X), 1)), np.diag((1 / sd_X))), axis=1)
        D = np.concatenate((D_1[np.newaxis, :], D_2))

        # Correct
        coef: NDArray[np.float64] = D @ beta_hat
        vcov: NDArray[np.float64] = D @ V0 @ D.T
        se: NDArray[np.float64] = np.sqrt(np.diagonal(D @ V0 @ D.T))

        self.result = ModelResult(coef, se, vcov)
        return self.result


class MeanEstimator:
    def __init__(self):
        """
        Initialize the MeanEstimator.
        """
        pass

    def fit(self, Y: NDArray[np.float64]) -> ModelResult:
        """
        Fit the mean estimator to the data.

        Args:
            Y (NDArray[np.float64]): Dependent variable

        Returns:
            ModelResult: The result of the model estimation
        """
        mean, se, vcov = _mean_estimator(Y)
        self.result = ModelResult(
            coef=np.array([mean]),
            se=np.array([se]),
            vcov=vcov
        )
        return self.result
    

