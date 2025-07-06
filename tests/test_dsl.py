"""
Testing the DSL estimators
"""

import pytest
import numpy as np
import pandas as pd
from dsl.dsl import DSLModel
from dsl.estimators import LogisticRegression, MeanEstimator, ModelResult
from dsl.helpers import _logistic, _scale
from sklearn.ensemble import RandomForestRegressor


# Create helper function for generating synthetic data
def generate_synthetic_logistic_regression_data(n=200, p=4, beta_true=None):
    """
    Generate synthetic data for logistic regression testing.
    """
    if beta_true is None:
        beta_true = np.random.randn(p)

    X = np.ones((n, p))  # Include intercept
    X[:, 1:] = np.random.randn(n, p - 1)  # Random continuous variables
    logits = X @ beta_true
    probs = _logistic(logits)
    Y = np.random.binomial(1, probs).astype(np.float64)  # Binary outcome
    return Y, X, beta_true


def generate_synthetic_mean_data(n=100, mean=5, std=2):
    """
    Generate synthetic data for mean estimation testing.
    """
    Y = np.random.normal(mean, std, n)
    return Y


def generate_binary_surrogates(
    Y: np.ndarray, X: np.ndarray | None, noise_scale=0.2
) -> np.ndarray:
    """
    Generate binary surrogates for logistic regression testing.
    If X is provided, the flip probability depends on X.
    The proportion of flipped values is controlled by noise_scale.
    """
    n = len(Y)
    # Generate base flip probabilities
    if X is not None:
        # Use X to create spatially correlated flip probabilities
        # Create a simple linear combination of X features (excluding intercept if present)
        X_features = X[:, 1:] if X.shape[1] > 1 else X
        weights = np.random.randn(X_features.shape[1]) * 0.1
        flip_logits = X_features @ weights
        base_flip_probs = _logistic(flip_logits)
        # Scale to desired noise level
        flip_probs = base_flip_probs * noise_scale
    else:
        # Use constant flip probability
        flip_probs = np.full(n, noise_scale)

    # Generate binary surrogates by flipping some values
    flips = np.random.binomial(1, flip_probs)
    Y_surrogates = Y.copy()
    Y_surrogates[flips == 1] = 1 - Y_surrogates[flips == 1]  # Flip binary values

    return Y_surrogates.astype(np.float64)


def generate_continuous_surrogates(
    Y: np.ndarray, X: np.ndarray | None, bias=0.2, noise_scale=0.2
) -> np.ndarray:
    """
    Generate continuous surrogates for mean estimation testing.
    If X is provided, the noise depends on X.
    `noise_scale` is the size of the bias relative to the true mean.
    """
    bias = bias * np.mean(Y)  # Scale noise relative to mean of Y
    n = len(Y)
    # Generate noise based on X if provided
    if X is not None:
        # Use X to create spatially correlated noise
        # Create a simple linear combination of X features (excluding intercept if present)
        X_features = X[:, 1:] if X.shape[1] > 1 else X
        weights = np.random.randn(X_features.shape[1]) * 0.1
        noise_logits = X_features @ weights
        noise_probs = _logistic(noise_logits)
        noise = np.random.normal(bias, noise_probs * noise_scale, n)
    else:
        # Use constant noise level
        noise = np.random.normal(bias, noise_scale, n)

    Y_surrogates = Y + noise
    return Y_surrogates.astype(np.float64)


# Create fixtures
@pytest.fixture(scope="module")
def generate_logistic_data_fixture():
    """
    Fixture to generate synthetic logistic regression data.
    """
    Y, X, beta_true = generate_synthetic_logistic_regression_data()
    Q = generate_binary_surrogates(Y, X)
    return Y, X, beta_true, Q


@pytest.fixture(scope="module")
def generate_mean_data_fixture():
    """
    Fixture to generate synthetic mean estimation data.
    """
    Y = generate_synthetic_mean_data()
    Q = generate_continuous_surrogates(Y, None)
    return Y, Q


# End-to-end test cases for DSL estimators
def test_dsl_mean_estimation_default_args(generate_mean_data_fixture):
    """
    Test complete mean estimation workflow from data generation to estimation.
    """
    Y, Q = generate_mean_data_fixture

    # Get ground truth
    true_me = MeanEstimator()
    true_result = true_me.fit(Y)

    # Fit naive estimator
    naive_me = MeanEstimator()
    naive_result = naive_me.fit(Q)

    # Fit DSL model
    data = pd.DataFrame({"Y": Y, "Q": Q})
    dsl_model = DSLModel(
        data=data, gold_standard_col="Y", predicted_val_cols="Q", model_type="mean",
    )
    dsl_result = dsl_model.fit()

    # Verify result structure
    assert isinstance(dsl_result, ModelResult)
    assert len(dsl_result.coef) == 1
    assert len(dsl_result.se) == 1

    # Check that DSL is closer to true mean than naive estimator
    assert (
        true_result.coef[0] - dsl_result.coef[0]
        < true_result.coef[0] - naive_result.coef[0]
    ), "DSL estimator should be closer to true mean than naive estimator"


def test_dsl_logistic_regression_default_args(generate_logistic_data_fixture):
    """
    Test complete logistic regression workflow from data generation to estimation.
    """
    Y, X, beta_true, Q = generate_logistic_data_fixture

    # Fit true logistic regression
    true_lr = LogisticRegression()
    true_result = true_lr.fit(Y, X)

    # Fit naive estimator
    naive_lr = LogisticRegression()
    naive_result = naive_lr.fit(Q, X)

    # Fit DSL model
    X_cols = {f"X{i}": X[:, i] for i in range(X.shape[1])}
    data = pd.DataFrame({"Y": Y, "Q": Q} | X_cols)
    dsl_model = DSLModel(
        data=data,
        gold_standard_col="Y",
        predicted_val_cols="Q",
        regressor_cols=list(X_cols.keys()),
        model_type="logistic",
    )
    dsl_result = dsl_model.fit()

    # Verify result structure
    assert isinstance(dsl_result, ModelResult)
    assert len(dsl_result.coef) == X.shape[1]
    assert len(dsl_result.se) == X.shape[1]

    # Check that DSL is closer to true coefficients than naive estimator
    for i in range(len(beta_true)):
        assert (
            abs(true_result.coef[i] - dsl_result.coef[i]) <
            abs(true_result.coef[i] - naive_result.coef[i])
        ), f"DSL estimator should be closer to true coefficient {i} than naive estimator"
