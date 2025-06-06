"""
MoM implementation of multivariate logistic regression

Authors:
  - Musashi Hinck

Steps:
------
1. Prepare inputs to optimizer
    - objective function
    - function arguments
    - initial guess for parameters
    - jacobian (i.e. derivative of objective function with respect to parameters)
2. Run optimizer
3. Compute standard errors
4. Recalibrate using robust standard errors

Notes:
- TODO: Add formula support (analogous to statsmodels)

"""


from collections import namedtuple
import numpy as np
from typing import Optional
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from numba import njit

ModelResult = namedtuple("ModelResult", ["coef", "se", "vcov"])


@njit
def logistic(r: ArrayLike) -> NDArray[np.float64]:
    "JIT logistic function"
    return 1 / (1 + np.exp(-r))


@njit
def _logit_objective(Y, X, beta, lam=1e-5) -> float:
    inv_logit = logistic(X @ beta)
    m = X.T @ ((Y - inv_logit) / X.shape[0])
    return np.mean(m**2) + lam * np.mean(beta**2)


def logit_objective(
    Y: NDArray[np.float64],  # y
    X: NDArray[np.float64],  # x
    beta: NDArray[np.float64],  # beta
    lam: Optional[float] = 1e-5,  # lambda
) -> float:
    """
    Logistic regression objective function wrapper, uses _logit_objective JIT compiled function.

    Args:
        Y (NDArray[np.float64]): Dependent variable
        X (NDArray[np.float64]): Independent variables
        beta (NDArray[np.float64]): Beta coefficients
        lam (Optional[float]): Regularization parameter

    Returns:
        float: Objective function output

    """

    # Checks
    assert (
        Y.shape[0] == X.shape[0]
    ), "Outcome and inputs have different number of observations"
    assert (
        X[:, 0] == np.ones(X.shape[0])
    ).all(), "We assume that inputs have a column of ones (i.e. a constant term)"
    assert (
        beta.shape[0] == X.shape[1]
    ), "Mismatching number of parameters to fit and input features"

    # Call compiled function
    return _logit_objective(Y, X, beta, lam)


@njit
def logit_jacobian(
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
    inv_logit = logistic(X @ beta)
    A = (Y - inv_logit).T @ X
    B = (-(inv_logit) * (1 - inv_logit)) @ X2
    out = cons * A * B + lam * np.mean(2 * np.abs(beta))
    return out


@njit
def numbastd(x: ArrayLike) -> NDArray[np.float64]:
    "JIT sample standard deviation with ddof=1"
    return np.sqrt(((x - x.mean()) ** 2).sum() / (x.shape[0] - 1))


@njit
def _vecscale(x: ArrayLike) -> NDArray[np.float64]:
    xmean = x.mean()
    xstd = numbastd(x)
    return (x - xmean) / xstd


@njit
def _scale(X: NDArray):
    # Initialize an empty matrix to store the results
    result = np.empty_like(X)
    for i in range(X.shape[1]):  # Iterate over columns
        result[:, i] = _vecscale(X[:, i])
    return result


# Reformat for minimize
def _obj(x, *args):
    Y, X_scaled, _, _, lam = [*args]
    beta = x
    return logit_objective(Y, X_scaled, beta, lam)


def _jac(x, *args):
    Y, X, X2, cons, lam = [*args]
    beta = x
    return logit_jacobian(Y, X, X2, beta, cons, lam)


def logistic_regression(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    lam: Optional[float] = 1e-5,
    show_optimizer_trace: Optional[bool] = False,
) -> "ModelResult":
    """Method of Moments Logistic Regression"""
    # Pre-compute inputs to minimizer
    num_obs: int = X.shape[0]
    num_covariates: int = X.shape[1]
    x0: NDArray = np.zeros((num_covariates,))
    X_scaled: NDArray = np.concatenate(
        (np.ones((num_obs, 1)), _scale(X[:, 1:])), axis=1
    )
    cons: int = 2 / (num_covariates * num_obs**2)
    X_scaled_squared: NDArray = X_scaled**2

    # Optimize
    optimizer_output = minimize(
        fun=_obj,
        x0=x0,
        args=(Y, X_scaled, X_scaled_squared, cons, lam),
        jac=_jac,
        method="L-BFGS-B",
        options={"maxiter": 1000, "disp": show_optimizer_trace},
    )

    # Collect
    beta_hat: NDArray[np.float64] = optimizer_output.x

    # Standard Error
    pi = logistic(X_scaled @ beta_hat)
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

    return ModelResult(coef, se, vcov)