"""
Time-Series Dataset Builder for Quantitative Alpha Models.
Implements purged time-series splitting to guarantee zero lookahead bias.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from microquant.core.logging import get_logger
from microquant.core.exceptions import DataValidationError

logger = get_logger("microquant.dataset")


@dataclass
class DatasetSplits:
    """Holds train and test data splits with metadata."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    timestamps_train: pd.Series
    timestamps_test: pd.Series
    prices_test: pd.Series
    horizon: int


class AlphaDatasetBuilder:
    """
    Constructs high-frequency alpha prediction targets and leak-free train/test splits.
    """

    FEATURE_SUBSET: List[str] = [
        "spread_bps",
        "order_imbalance_1",
        "depth_imbalance_3",
        "weighted_imbalance",
        "micro_price_bias",
        "bid_slope",
        "ask_slope",
        "slope_ratio",
        "signed_quantity",
        "price_to_vwap_bps",
        "effective_spread",
        "return_1",
        "log_return_1",
        "volatility_10",
        "volatility_20",
        "ma_ratio",
        "rsi_14",
        "bb_pct_b",
    ]

    def __init__(self, horizon: int = 10, train_ratio: float = 0.75):
        self.horizon = horizon
        self.train_ratio = train_ratio

    def build_tick_dataset(self, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, DatasetSplits]:
        """
        Creates forward-looking target and purged time-series splits from tick features.

        Target: 1 if mid-price at (t + horizon) > mid-price at t, else 0.
        Purging: Drops the horizon ticks before the test split boundary to prevent label leakage.
        """
        logger.info(f"Building alpha dataset with forward horizon = {self.horizon} ticks...")
        df = df_features.copy()

        if "mid_price" not in df.columns:
            raise DataValidationError("DataFrame missing 'mid_price' column required for target generation.")

        # Forward mid price target
        future_mid = df["mid_price"].shift(-self.horizon)
        df["target"] = (future_mid > df["mid_price"]).astype(float)
        # Last `horizon` rows do not have forward target -> drop them
        df = df.iloc[:-self.horizon].copy()

        # Select available features
        features = [col for col in self.FEATURE_SUBSET if col in df.columns]
        if not features:
            raise DataValidationError(f"None of the required features found in DataFrame columns: {df.columns.tolist()}")

        logger.info(f"Using {len(features)} microstructure features for modeling.")

        # Strict chronological split
        total_rows = len(df)
        split_idx = int(total_rows * self.train_ratio)

        # Purging: Drop horizon rows right before split_idx so train target does not leak into test features
        train_end_idx = max(0, split_idx - self.horizon)
        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(
            f"Split data: Train={len(train_df)} rows, Test={len(test_df)} rows. "
            f"Purged {split_idx - train_end_idx} rows at boundary."
        )

        X_train_raw = train_df[features].values
        y_train = train_df["target"].values.astype(int)

        X_test_raw = test_df[features].values
        y_test = test_df["target"].values.astype(int)

        # Fit scaler ONLY on train set
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        splits = DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=features,
            scaler=scaler,
            timestamps_train=train_df["timestamp"] if "timestamp" in train_df.columns else pd.Series(dtype=object),
            timestamps_test=test_df["timestamp"] if "timestamp" in test_df.columns else pd.Series(dtype=object),
            prices_test=test_df["price"] if "price" in test_df.columns else test_df["mid_price"],
            horizon=self.horizon,
        )

        return df, splits
