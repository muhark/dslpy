"""
Tests for dsl.helpers module
"""

import pytest
import numpy as np
from dsl.helpers import (
    _numbastd,
    _vecscale,
    _scale,
    _logistic,
    _confidence_intervals,
    _logit_objective,
    _logit_jacobian,
    _mean_estimator,
)


def test_numbastd_matches_numpy():
    """
    Test that numbastd matches numpy's std with ddof=1.
    """
    x = np.random.randn(100)
    assert np.allclose(_numbastd(x), np.std(x, ddof=1), atol=1e-8)


def test_vecscale_mean_std():
    """
    Test that _vecscale produces mean 0 and std 1.
    """
    x = np.random.randn(100)
    scaled = _vecscale(x)
    assert np.allclose(scaled.mean(), 0, atol=1e-8)
    assert np.allclose(np.std(scaled, ddof=1), 1, atol=1e-8)


def test_scale_matrix():
    """
    Test that _scale produces standardized columns.
    """
    X = np.random.randn(10, 3)
    X_scaled = _scale(X)
    assert X_scaled.shape == X.shape
    for i in range(X.shape[1]):
        assert np.allclose(X_scaled[:, i].mean(), 0, atol=1e-8)
        assert np.allclose(np.std(X_scaled[:, i], ddof=1), 1, atol=1e-8)


def test_logistic():
    """
    Test the logistic function with a range of values.
    """
    x = np.array([-5, -1, 0, 1, 5], dtype=np.float64)
    out = _logistic(x)
    expected = 1 / (1 + np.exp(-x))
    assert np.allclose(out, expected, atol=1e-6)


def test_logistic_extreme_values():
    """
    Test the logistic function with extreme values to ensure it handles them correctly.
    """
    x = np.array([
        np.finfo(np.float64).min,
        0,
        np.finfo(np.float64).max
    ], dtype=np.float64)
    out = _logistic(x)
    assert np.allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


def test_logit_objective_shapes():
    """
    Test that logit_objective returns correct shapes and types.
    """
    Y = np.array([0, 1, 1, 0], dtype=np.float64)
    X = np.ones((4, 2), dtype=np.float64)
    X[:, 1] = np.arange(4)
    beta = np.array([0.1, -0.2], dtype=np.float64)
    # Add constant column
    assert X[:, 0].all() == 1
    val = _logit_objective(Y, X, beta)
    assert isinstance(val, float)


def test_logit_jacobian_output_shape():
    """
    Test that logit_jacobian returns correct output shape.
    """
    Y = np.array([0, 1, 1, 0], dtype=np.float64)
    X = np.ones((4, 2), dtype=np.float64)
    X[:, 1] = np.arange(4)
    X2 = X ** 2
    beta = np.array([0.1, -0.2], dtype=np.float64)
    out = _logit_jacobian(Y, X, X2, beta)
    assert out.shape == beta.shape


def test_confidence_intervals_shape():
    """
    Test that _confidence_intervals returns correct shape.
    """
    coef = np.array([1.0, 2.0, 3.0])
    se = np.array([0.1, 0.2, 0.3])
    ci = _confidence_intervals(coef, se)
    assert ci.shape == (3, 2)
    # Lower bounds should be less than coefficients
    assert np.all(ci[:, 0] < coef)
    # Upper bounds should be greater than coefficients
    assert np.all(ci[:, 1] > coef)


def test_mean_estimator():
    """
    Test that _mean_estimator returns correct statistics.
    """
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, se, vcov = _mean_estimator(Y)
    
    # Check mean
    assert np.allclose(mean, 3.0)
    
    # Check standard error
    expected_se = np.std(Y, ddof=1) / np.sqrt(len(Y))
    assert np.allclose(se, expected_se)
    
    # Check variance-covariance matrix
    assert vcov.shape == (1, 1)
    assert np.allclose(vcov[0, 0], se**2)

