"""
tests/test_pipeline.py
-----------------------
Unit tests covering the parts of the pipeline most likely to silently
break things or leak data: feature engineering shape/NaN handling,
and the walk-forward split logic (no overlap, no look-ahead).
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data
from src.features import build_features, FEATURE_COLUMNS
from src.train import walk_forward_splits


@pytest.fixture
def sample_raw_data():
    return generate_synthetic_data(start="2020-01-01", end="2022-12-31", seed=1)


def test_synthetic_data_shape(sample_raw_data):
    assert not sample_raw_data.empty
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in sample_raw_data.columns
    assert (sample_raw_data["High"] >= sample_raw_data["Low"]).all()
    assert (sample_raw_data["Close"] > 0).all()


def test_build_features_no_nans(sample_raw_data):
    feat_df = build_features(sample_raw_data)
    assert not feat_df.isnull().values.any(), "Feature dataframe should have NaNs dropped"
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns
    assert set(feat_df["target"].unique()).issubset({0, 1})


def test_build_features_smaller_than_raw(sample_raw_data):
    feat_df = build_features(sample_raw_data)
    # rolling windows + the shifted label should shrink the usable dataset
    assert len(feat_df) < len(sample_raw_data)


def test_walk_forward_no_lookahead():
    """The core correctness property: every test index must come strictly
    after every train index within the same fold."""
    n_rows = 1000
    for train_idx, test_idx in walk_forward_splits(n_rows, n_splits=4, min_train_size=300):
        assert train_idx.max() < test_idx.min(), "Test fold must be strictly after training data"
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test indices must not overlap"


def test_walk_forward_expanding_window():
    """Each successive fold's training set should be larger (expanding window)."""
    n_rows = 1000
    train_sizes = [len(train_idx) for train_idx, _ in walk_forward_splits(n_rows, n_splits=4, min_train_size=300)]
    assert train_sizes == sorted(train_sizes), "Training window should expand across folds"


def test_walk_forward_raises_on_insufficient_data():
    with pytest.raises(ValueError):
        list(walk_forward_splits(n_rows=100, n_splits=5, min_train_size=500))
