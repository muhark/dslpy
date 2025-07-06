"""
Data utilities for Design-based Supervised Learning model.
Authors:
  - Musashi Hinck
"""


from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class DSLDatasetColumns:
    """
    DSLDatasetColumns is a dataclass that holds the column names for the DSLDataset.
    It is used to ensure consistent access to the columns in the dataset.
    """

    gold_standard_col: str
    predicted_val_cols: str | list[str]
    regressor_cols: str | list[str] = field(default=None)
    predictor_cols: str | list[str] = field(default=None)
    sample_weights: str = field(default=None)
    missing_indicator: str = field(default=None)

    def __post_init__(self):
        # At a minimum we need gold_standard_col and predicted_val_cols
        assert self.gold_standard_col is not None, "gold_standard_col must be specified"
        assert self.predicted_val_cols is not None, "predicted_val_cols must be specified"

        # Ensure gold_standard_col is a string
        assert isinstance(self.gold_standard_col, str), "gold_standard_col must be a string"

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
        if self.regressor_cols is not None:
            assert all(
                isinstance(col, str) for col in self.regressor_cols
            ), "regressor_cols must be a string or list of strings"
        if self.predictor_cols is not None:
            assert all(
                isinstance(col, str) for col in self.predictor_cols
            ), "predictor_cols must be a string or list of strings"

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
        if self.regressor_cols is not None:
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
        regressor_cols: str | list[str] = None,  # X
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
            if self._col_ref.regressor_cols
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
    
    def __len__(self) -> int:
        """
        Return the number of rows in the dataset.
        """
        return len(self.dataframe)

    @property
    def QWX(self) -> np.ndarray:
        return np.hstack(
            [arr for arr in [self.Q, self.W, self.X] if arr is not None]
        )
