"""
TODO: Compare results with statsmodels
TODO: Compare results with sklearn
"""

import pytest
import numpy as np
from dsl.logit import (
    logistic,
    logit_objective,
    logit_jacobian,
    numbastd,
    _vecscale,
    _scale,
    logistic_regression,
    ModelResult,
)

def test_logistic_basic():
    """
    Test the logistic function with extreme values to ensure it handles them correctly.
    """
    x = np.array([
        np.finfo(np.float64).min,
        0,
        np.finfo(np.float64).max
    ], dtype=np.float64)
    out = logistic(x)
    assert np.allclose(out, [0.0, 0.5, 1.0], atol=1e-6)

def test_logit_objective_shapes():
    Y = np.array([0, 1, 1, 0], dtype=np.float64)
    X = np.ones((4, 2), dtype=np.float64)
    X[:, 1] = np.arange(4)
    beta = np.array([0.1, -0.2], dtype=np.float64)
    # Add constant column
    assert X[:, 0].all() == 1
    val = logit_objective(Y, X, beta)
    assert isinstance(val, float)

def test_logit_objective_asserts():
    Y = np.array([0, 1, 1, 0], dtype=np.float64)
    X = np.ones((4, 2), dtype=np.float64)
    beta = np.array([0.1, -0.2], dtype=np.float64)
    # X missing constant
    X[:, 0] = 2
    with pytest.raises(AssertionError):
        logit_objective(Y, X, beta)
    # Mismatched shapes
    X = np.ones((4, 3), dtype=np.float64)
    X[:, 0] = 1
    with pytest.raises(AssertionError):
        logit_objective(Y, X, beta)

def test_logit_jacobian_output_shape():
    Y = np.array([0, 1, 1, 0], dtype=np.float64)
    X = np.ones((4, 2), dtype=np.float64)
    X[:, 1] = np.arange(4)
    X2 = X ** 2
    beta = np.array([0.1, -0.2], dtype=np.float64)
    out = logit_jacobian(Y, X, X2, beta)
    assert out.shape == beta.shape

def test_numbastd_matches_numpy():
    x = np.random.randn(100)
    assert np.allclose(numbastd(x), np.std(x, ddof=1), atol=1e-8)

def test_vecscale_mean_std():
    x = np.random.randn(100)
    scaled = _vecscale(x)
    assert np.allclose(scaled.mean(), 0, atol=1e-8)
    assert np.allclose(np.std(scaled, ddof=1), 1, atol=1e-8)

def test_scale_matrix():
    X = np.random.randn(10, 3)
    X_scaled = _scale(X)
    assert X_scaled.shape == X.shape
    for i in range(X.shape[1]):
        assert np.allclose(X_scaled[:, i].mean(), 0, atol=1e-8)
        assert np.allclose(np.std(X_scaled[:, i], ddof=1), 1, atol=1e-8)

def test_logistic_regression_output():
    np.random.seed(42)
    n = 100
    X = np.ones((n, 3))
    X[:, 1:] = np.random.randn(n, 2)
    beta_true = np.array([0.5, -1.0, 2.0])
    logits = X @ beta_true
    p = 1 / (1 + np.exp(-logits))
    Y = np.random.binomial(1, p)
    result = logistic_regression(Y, X)
    assert isinstance(result, ModelResult)
    assert result.coef.shape[0] == X.shape[1]
    assert result.se.shape[0] == X.shape[1]
    assert result.vcov.shape == (X.shape[1], X.shape[1])
    # Coefficient signs should match true beta
    assert np.sign(result.coef[1]) == np.sign(beta_true[1])
    assert np.sign(result.coef[2]) == np.sign(beta_true[2])
