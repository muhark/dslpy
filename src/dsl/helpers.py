"""
Numba-optimized functions for statistical calculations.
Authors:
   - Musashi Hinck 

"""

import numpy as np
from typing import Optional
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from numba import njit


@njit
def _numbastd(x: ArrayLike) -> NDArray[np.float64]:
    "JIT sample standard deviation with ddof=1"
    return np.sqrt(((x - x.mean()) ** 2).sum() / (x.shape[0] - 1))


@njit
def _vecscale(x: ArrayLike) -> NDArray[np.float64]:
    xmean = x.mean()
    xstd = _numbastd(x)
    return (x - xmean) / xstd


@njit
def _scale(X: NDArray):
    # Initialize an empty matrix to store the results
    result = np.empty_like(X)
    for i in range(X.shape[1]):  # Iterate over columns
        result[:, i] = _vecscale(X[:, i])
    return result


@njit
def _logistic(r: ArrayLike) -> NDArray[np.float64]:
    "JIT logistic function"
    return 1 / (1 + np.exp(-r))


def _zscore(alpha: float = 0.05) -> float:
    """
    Function to calculate the z-score for a given alpha level.
    
    Args:
        alpha (float): Significance level (default: 0.05)

    Returns:
        float: z-score corresponding to the alpha level
    """
    return np.abs(np.percentile(np.random.normal(size=10000), 100 * (1 - alpha / 2)))


def _confidence_intervals(coef: NDArray[np.float64], se: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.float64]:
    """
    Function to calculate confidence intervals for coefficients.
    
    Args:
        coef (NDArray[np.float64]): Coefficients of the model
        se (NDArray[np.float64]): Standard errors of the coefficients
        alpha (float): Significance level for the confidence interval (default: 0.05)

    Returns:
        NDArray[np.float64]: Confidence intervals for the coefficients
    """
    z_score = _zscore(alpha)
    lower_bound = coef - z_score * se
    upper_bound = coef + z_score * se
    return np.column_stack((lower_bound, upper_bound))


@njit
def _logit_objective(Y, X, beta, lam=1e-5) -> float:
    inv_logit = _logistic(X @ beta)
    m = X.T @ ((Y - inv_logit) / X.shape[0])
    return np.mean(m**2) + lam * np.mean(beta**2)


@njit
def _logit_jacobian(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    X2: NDArray[np.float64],
    beta: NDArray[np.float64],
    cons: Optional[float] = None,
    lam: Optional[float] = 1e-5,
) -> NDArray:
    "JIT logit Jacobian (i.e. first derivative of objective function)"
    if cons is None:
        cons = 2 / (X.shape[1] * X.shape[0] ** 2)
    inv_logit = _logistic(X @ beta)
    A = (Y - inv_logit).T @ X
    B = (-(inv_logit) * (1 - inv_logit)) @ X2
    out = cons * A * B + lam * np.mean(2 * np.abs(beta))
    return out

# Mean estimator
@njit
def _mean_estimator(Y: np.ndarray) -> tuple[np.float64, np.float64, np.ndarray]:
    """
    JIT compiled function to estimate the mean of a single variable. 
    """
    mean = np.mean(Y)
    se = _numbastd(Y) / np.sqrt(len(Y))
    vcov = np.array([[se ** 2]])
    return mean, se, vcov