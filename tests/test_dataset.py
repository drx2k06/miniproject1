"""
Unit Tests for Alpha Dataset Builder.
"""

import pytest
import numpy as np
import pandas as pd
from microquant.pipeline.features import MicrostructureFeatureExtractor
from microquant.models.dataset import AlphaDatasetBuilder


def test_purged_dataset_builder(sample_clean_trades):
    fe = MicrostructureFeatureExtractor()
    df_feat = fe.extract(sample_clean_trades, dropna=True)

    horizon = 5
    builder = AlphaDatasetBuilder(horizon=horizon, train_ratio=0.7)
    df_ml, splits = builder.build_tick_dataset(df_feat)

    assert len(splits.X_train) > 0
    assert len(splits.X_test) > 0
    assert len(splits.y_train) == len(splits.X_train)
    assert len(splits.y_test) == len(splits.X_test)

    # Labels must be binary {0, 1}
    assert set(np.unique(splits.y_train)).issubset({0, 1})
    assert set(np.unique(splits.y_test)).issubset({0, 1})

    # Scaler must have been fitted on train
    assert splits.scaler.mean_ is not None
    assert len(splits.feature_names) == splits.X_train.shape[1]
