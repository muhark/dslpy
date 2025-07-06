"""
Tests for dsl.estimators module
"""

import pytest
import numpy as np
from dsl.estimators import (
    LogisticRegression,
    MeanEstimator,
    ModelResult,
)


def test_model_result_creation():
    """
    Test that ModelResult can be created and accessed properly.
    """
    coef = np.array([1.0, 2.0])
    se = np.array([0.1, 0.2])
    vcov = np.array([[0.01, 0.005], [0.005, 0.04]])
    
    result = ModelResult(coef=coef, se=se, vcov=vcov)
    
    assert np.array_equal(result.coef, coef)
    assert np.array_equal(result.se, se)
    assert np.array_equal(result.vcov, vcov)


def test_model_result_confidence_intervals():
    """
    Test that ModelResult.confidence_intervals works correctly.
    """
    coef = np.array([1.0, 2.0])
    se = np.array([0.1, 0.2])
    vcov = np.array([[0.01, 0.005], [0.005, 0.04]])
    
    result = ModelResult(coef=coef, se=se, vcov=vcov)
    ci = result.confidence_intervals(alpha=0.05)
    
    assert ci.shape == (2, 2)
    # Lower bounds should be less than coefficients
    assert np.all(ci[:, 0] < coef)
    # Upper bounds should be greater than coefficients
    assert np.all(ci[:, 1] > coef)


def test_logistic_regression_initialization():
    """
    Test that LogisticRegression can be initialized with different parameters, including minimizer_options.
    """
    # Default initialization
    lr1 = LogisticRegression()
    assert lr1.lam == 1e-5
    assert lr1.minimizer_options == {"maxiter": 1000}

    # Custom initialization
    lr2 = LogisticRegression(lam=1e-3)
    assert lr2.lam == 1e-3

    # Custom minimizer_options
    options = {"maxiter": 200}
    lr3 = LogisticRegression(lam=1e-2, minimizer_options=options)
    assert lr3.lam == 1e-2
    assert lr3.minimizer_options == options

def test_logistic_regression_fit_output():
    """
    Test that LogisticRegression.fit returns correct output structure.
    """
    np.random.seed(42)
    n = 100
    X = np.ones((n, 3))
    X[:, 1:] = np.random.randn(n, 2)
    beta_true = np.array([0.5, -1.0, 2.0])
    logits = X @ beta_true
    p = 1 / (1 + np.exp(-logits))
    Y = np.random.binomial(1, p).astype(np.float64)
    
    lr = LogisticRegression()
    result = lr.fit(Y, X)
    
    assert isinstance(result, ModelResult)
    assert result.coef.shape[0] == X.shape[1]
    assert result.se.shape[0] == X.shape[1]
    assert result.vcov.shape == (X.shape[1], X.shape[1])
    
    # Check that the estimator stores the result
    assert hasattr(lr, 'result')
    assert lr.result is result


def test_logistic_regression_coefficient_signs():
    """
    Test that LogisticRegression recovers approximately correct coefficient signs.
    """
    np.random.seed(42)
    n = 100
    X = np.ones((n, 3))
    X[:, 1:] = np.random.randn(n, 2)
    beta_true = np.array([0.5, -1.0, 2.0])
    logits = X @ beta_true
    p = 1 / (1 + np.exp(-logits))
    Y = np.random.binomial(1, p).astype(np.float64)
    
    lr = LogisticRegression()
    result = lr.fit(Y, X)
    
    # Coefficient signs should match true beta (at least for non-intercept terms)
    assert np.sign(result.coef[1]) == np.sign(beta_true[1])
    assert np.sign(result.coef[2]) == np.sign(beta_true[2])


def test_mean_estimator_initialization():
    """
    Test that MeanEstimator can be initialized.
    """
    me = MeanEstimator()
    assert hasattr(me, 'fit')


def test_mean_estimator_fit():
    """
    Test that MeanEstimator.fit works correctly.
    """
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    me = MeanEstimator()
    result = me.fit(Y)
    
    assert isinstance(result, ModelResult)
    assert result.coef.shape == (1,)
    assert result.se.shape == (1,)
    assert result.vcov.shape == (1, 1)
    
    # Check that mean is correct
    assert np.allclose(result.coef[0], 3.0)
    
    # Check that standard error is correct
    expected_se = np.std(Y, ddof=1) / np.sqrt(len(Y))
    assert np.allclose(result.se[0], expected_se)
    
    # Check that the estimator stores the result
    assert hasattr(me, 'result')
    assert me.result is result


def test_logistic_regression_with_regularization():
    """
    Test that LogisticRegression works with different regularization parameters.
    """
    np.random.seed(42)
    n = 50
    X = np.ones((n, 2))
    X[:, 1] = np.random.randn(n)
    beta_true = np.array([0.0, 1.0])
    logits = X @ beta_true
    p = 1 / (1 + np.exp(-logits))
    Y = np.random.binomial(1, p).astype(np.float64)
    
    # Test with different regularization
    lr1 = LogisticRegression(lam=0.0)
    lr2 = LogisticRegression(lam=1.0)
    
    result1 = lr1.fit(Y, X)
    result2 = lr2.fit(Y, X)
    
    # Both should return valid results
    assert isinstance(result1, ModelResult)
    assert isinstance(result2, ModelResult)
    
    # Results should be different due to regularization
    assert not np.allclose(result1.coef, result2.coef)


def test_logistic_regression_input_validation():
    """
    Test that LogisticRegression handles input validation properly.
    """
    lr = LogisticRegression()
    
    # Test with mismatched dimensions
    Y = np.array([0, 1, 1], dtype=np.float64)
    X = np.ones((4, 2), dtype=np.float64)  # Different number of rows
    
    with pytest.raises((AssertionError, ValueError)):
        lr.fit(Y, X)

