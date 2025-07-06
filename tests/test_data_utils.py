"""
Tests for dsl.data_utils module
"""

import pytest
import numpy as np
import pandas as pd
from dsl.data_utils import DSLDatasetColumns, DSLDataset

from dsl.data_utils import (
    DSLDatasetColumns,
    DSLDataset
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "y": [1.0, 2.0, np.nan, 4.0],
        "Q1": [0.9, 2.1, 3.0, 4.2],
        "Q2": [1.1, 1.9, 2.8, 3.9],
        "X1": [10, 20, 30, 40],
        "X2": [5, 15, 25, 35],
        "W1": [100, 200, 300, 400],
        "pi": [0.1, 0.2, 0.3, 0.4],
        "R": [1, 1, 0, 1]
    })

def test_DSLDatasetColumns_str_list(sample_df):
    # Test with string columns
    cols = DSLDatasetColumns(
        gold_standard_col="y",
        predicted_val_cols="Q1",
    )
    assert cols.predicted_val_cols == ["Q1"]
    cols.check_columns(sample_df)

    # Test with list columns
    cols = DSLDatasetColumns(
        gold_standard_col="y",
        predicted_val_cols=["Q1", "Q2"],
        regressor_cols=["X1"],
        predictor_cols="W1",
        sample_weights="pi",
        missing_indicator="R"
    )
    assert cols.predicted_val_cols == ["Q1", "Q2"]
    assert cols.regressor_cols == ["X1"]
    assert cols.predictor_cols == ["W1"]
    cols.check_columns(sample_df)

def test_DSLDatasetColumns_str_list(sample_df):
    # Test with string and list columns
    cols = DSLDatasetColumns(
        gold_standard_col="y",
        predicted_val_cols=["Q1", "Q2"],
        regressor_cols="X1",
        predictor_cols=["W1"],
        sample_weights="pi",
        missing_indicator="R"
    )
    assert cols.predicted_val_cols == ["Q1", "Q2"]
    assert cols.regressor_cols == ["X1"]
    assert cols.predictor_cols == ["W1"]
    cols.check_columns(sample_df)


def test_DSLDatasetColumns_invalid_type():
    # Test with invalid type for predicted_val_cols
    with pytest.raises(TypeError):
        DSLDatasetColumns(
            gold_standard_col="y",
            predicted_val_cols=123,  # Invalid type
            regressor_cols="X1"
        )    


def test_DSLDatasetColumns_missing_column(sample_df):
    cols = DSLDatasetColumns(
        gold_standard_col="y",
        predicted_val_cols="Q1",
        regressor_cols="X1",
        predictor_cols="W1",
        sample_weights="pi",
        missing_indicator="R"
    )
    bad_df = sample_df.drop(columns=["Q1"])
    with pytest.raises(AssertionError):
        cols.check_columns(bad_df)


def test_DSLDataset_basic(sample_df):
    ds = DSLDataset(
        data=sample_df,
        gold_standard_col="y",
        predicted_val_cols=["Q1", "Q2"],
        regressor_cols=["X1", "X2"],
        predictor_cols="W1",
        sample_weights="pi",
        missing_indicator="R"
    )
    assert isinstance(ds.y, np.ndarray)
    assert ds.y.shape == (4,)
    assert ds.Q.shape == (4, 2)
    assert ds.X.shape == (4, 2)
    assert ds.W.shape == (4, 1)
    assert ds.pi.shape == (4,)
    assert ds.R.shape == (4,)
    assert ds.random_seed == 634

def test_DSLDataset_default_weights_and_missing(sample_df):
    ds = DSLDataset(
        data=sample_df,
        gold_standard_col="y",
        predicted_val_cols="Q1",
        regressor_cols="X1"
    )
    assert np.allclose(ds.pi, np.repeat(0.25, 4))
    assert np.array_equal(ds.R, sample_df["y"].notna().values)

def test_DSLDataset_getitem(sample_df):
    ds = DSLDataset(
        data=sample_df,
        gold_standard_col="y",
        predicted_val_cols=["Q1", "Q2"],
        regressor_cols="X1",
        predictor_cols="W1"
    )
    ds2 = ds[:2]
    assert isinstance(ds2, DSLDataset)
    assert ds2.y.shape == (2,)
    assert ds2.Q.shape == (2, 2)
    assert ds2.X.shape == (2, 1)
    assert ds2.W.shape == (2, 1)
    assert ds2.random_seed == ds.random_seed

def test_DSLDataset_no_WX(sample_df):
    ds = DSLDataset(
        data=sample_df,
        gold_standard_col="y",
        predicted_val_cols=["Q1", "Q2"]
    )
    assert ds.W is None
    assert ds.QWX.shape == (len(sample_df), 2)  # Q1, Q2

def test_DSLDataset_QWX_shape(sample_df):
    ds = DSLDataset(
        data=sample_df,
        gold_standard_col="y",
        predicted_val_cols=["Q1", "Q2"],
        regressor_cols="X1",
        predictor_cols="W1"
    )
    assert ds.QWX.shape == (len(sample_df), 4)  # Q1, Q2, X1, W1
