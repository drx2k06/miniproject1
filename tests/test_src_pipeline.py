"""
Comprehensive Unit and Integration Tests for src/ Modules.

Author: Daniel Richard Xavier
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data_loader import DataLoader, DataValidationError
from src.cleaner import DataCleaner
from src.features import MicrostructureFeatureExtractor
from src.resampler import OHLCVResampler
from src.analysis import StatisticalAnalyzer
from src.visualization import VisualizationSuite


@pytest.fixture
def sample_dirty_df():
    """Generates a representative dirty high-frequency LOB dataset."""
    return pd.DataFrame({
        "trade_id": ["T1", "T2", "T3", "T4", "T5", "T1"],  # T1 duplicate
        "timestamp": [
            "2026-10-01 09:00:00.123",
            "01/10/2026 09:00:01",
            "1727773202",  # unix s
            "2026-10-01 09:00:03.000",
            "2026-10-01 09:00:04.000",
            "2026-10-01 09:00:00.123",
        ],
        "symbol": ["BTCUSDT", "BTC/USDT", "btcusdt", "BTC-USDT", "BTCUSDT", "BTCUSDT"],
        "side": ["buy", "sell", "NaN", "BUY", "Sell", "buy"],
        "price": ["$100,000.00", "100010.0", "10000000.0", "-50.0", "100005.0", "$100,000.00"],  # T3 is 100x, T4 is negative
        "quantity": ["0.1", "0.2", "0.5", "0.0", "0.3", "0.1"],  # T4 is 0.0
        "bid_price_1": ["$99,995.0", "100005.0", "99995.0", "99995.0", "100006.0", "$99,995.0"],  # T5 has crossed book (100006 > 100005)
        "bid_qty_1": ["1.0", "2.0", "1.5", "1.0", "1.0", "1.0"],
        "bid_price_2": ["99990.0", "100000.0", "99990.0", "99990.0", "100000.0", "99990.0"],
        "bid_qty_2": ["2.0", "3.0", "2.0", "2.0", "2.0", "2.0"],
        "bid_price_3": ["99985.0", "99995.0", "99985.0", "99985.0", "99995.0", "99985.0"],
        "bid_qty_3": ["3.0", "4.0", "3.0", "3.0", "3.0", "3.0"],
        "ask_price_1": ["$100,005.0", "100015.0", "100005.0", "100005.0", "100005.0", "$100,005.0"],
        "ask_qty_1": ["1.2", "1.8", "1.0", "1.0", "1.0", "1.2"],
        "ask_price_2": ["100010.0", "100020.0", "100010.0", "100010.0", "100010.0", "100010.0"],
        "ask_qty_2": ["2.2", "2.5", "2.0", "2.0", "2.0", "2.2"],
        "ask_price_3": ["100015.0", "100025.0", "100015.0", "100015.0", "100015.0", "100015.0"],
        "ask_qty_3": ["3.5", "3.0", "3.0", "3.0", "3.0", "3.5"],
    })


def test_data_loader_validation(sample_dirty_df):
    loader = DataLoader()
    report = loader.validate_schema(sample_dirty_df)
    assert report.is_valid is True
    assert report.total_rows == 6
    assert report.duplicate_rows == 1
    assert report.duplicate_trade_ids == 1


def test_data_loader_missing_columns():
    loader = DataLoader()
    df_incomplete = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    report = loader.validate_schema(df_incomplete, required_columns=["trade_id", "price"])
    assert report.is_valid is False
    assert len(report.validation_messages) > 0


def test_data_loader_empty_df():
    loader = DataLoader()
    df_empty = pd.DataFrame()
    report = loader.validate_schema(df_empty, required_columns=["trade_id"])
    assert report.is_valid is False


def test_data_cleaner_pipeline(sample_dirty_df):
    cleaner = DataCleaner(max_mid_gap_pct=0.02)
    df_clean, audit = cleaner.clean(sample_dirty_df)

    # Duplicates removed
    assert audit.duplicate_rows_removed == 1

    # T1 duplicate, T4 non-positive dropped, T5 crossed book dropped
    # T3 scale error corrected (10000000.0 / 100 -> 100000.0)
    assert audit.scale_anomalies_corrected == 1
    assert audit.non_positive_values_dropped == 1
    assert audit.crossed_books_dropped == 1

    # All symbols normalized
    assert (df_clean["symbol"] == "BTCUSDT").all()

    # All prices in valid range
    assert (df_clean["price"] < 150000.0).all()
    assert (df_clean["price"] > 50000.0).all()

    # Sides imputed
    assert df_clean["side"].isna().sum() == 0
    assert set(df_clean["side"].unique()).issubset({"buy", "sell"})


def test_cleaner_timestamp_parsing():
    cleaner = DataCleaner()
    # Unix seconds
    t1 = cleaner.parse_mixed_timestamp("1727773200")
    assert t1.year == 2024 or t1.year == 2026 or t1.year > 2020

    # UK DD/MM/YYYY
    t2 = cleaner.parse_mixed_timestamp("01/10/2026 14:30:00")
    assert t2.day == 1
    assert t2.month == 10
    assert t2.year == 2026

    # ISO-8601
    t3 = cleaner.parse_mixed_timestamp("2026-10-01 09:00:00.500")
    assert t3.hour == 9
    assert t3.microsecond == 500000


def test_feature_extractor():
    # Construct 50 valid clean rows
    n = 50
    df = pd.DataFrame({
        "trade_id": [f"T{i}" for i in range(n)],
        "timestamp": pd.date_range("2026-10-01 09:00:00", periods=n, freq="s", tz="UTC"),
        "symbol": ["BTCUSDT"] * n,
        "side": ["buy" if i % 2 == 0 else "sell" for i in range(n)],
        "price": [100000.0 + i * 0.5 for i in range(n)],
        "quantity": [0.1 + (i % 3) * 0.05 for i in range(n)],
        "bid_price_1": [99995.0 + i * 0.5 for i in range(n)],
        "bid_qty_1": [1.0 + (i % 2) * 0.5 for i in range(n)],
        "bid_price_2": [99990.0 + i * 0.5 for i in range(n)],
        "bid_qty_2": [2.0] * n,
        "bid_price_3": [99985.0 + i * 0.5 for i in range(n)],
        "bid_qty_3": [3.0] * n,
        "ask_price_1": [100005.0 + i * 0.5 for i in range(n)],
        "ask_qty_1": [1.2] * n,
        "ask_price_2": [100010.0 + i * 0.5 for i in range(n)],
        "ask_qty_2": [2.2] * n,
        "ask_price_3": [100015.0 + i * 0.5 for i in range(n)],
        "ask_qty_3": [3.2] * n,
    })

    fe = MicrostructureFeatureExtractor(rsi_period=14, vol_window=20)
    df_feat = fe.extract(df, dropna=True)

    # With ma_30 rolling window, 50 - 30 + 1 = 21 rows remain
    assert len(df_feat) == n - 30 + 1
    assert "spread" in df_feat.columns
    assert "micro_price" in df_feat.columns
    assert "depth_imbalance_3" in df_feat.columns
    assert "vwap" in df_feat.columns
    assert "volatility_20" in df_feat.columns

    # Spread should be 10.0
    assert np.isclose(df_feat["spread"].iloc[0], 10.0)


def test_resampler():
    n = 120  # 2 minutes of 1-second ticks
    df = pd.DataFrame({
        "trade_id": [f"T{i}" for i in range(n)],
        "timestamp": pd.date_range("2026-10-01 09:00:00", periods=n, freq="s", tz="UTC"),
        "symbol": ["BTCUSDT"] * n,
        "side": ["buy" if i % 2 == 0 else "sell" for i in range(n)],
        "price": [100000.0 + (i % 10) * 0.2 for i in range(n)],
        "quantity": [0.1] * n,
    })

    resampler = OHLCVResampler()
    candles = resampler.resample(df, timeframe="1min")

    assert len(candles) == 2
    assert "open" in candles.columns
    assert "high" in candles.columns
    assert "low" in candles.columns
    assert "close" in candles.columns
    assert "volume" in candles.columns
    assert np.isclose(candles["volume"].sum(), 12.0)


def test_statistical_analyzer():
    analyzer = StatisticalAnalyzer()
    
    # 1. ADF Test on stationary vs non-stationary series
    np.random.seed(42)
    stationary_s = pd.Series(np.random.normal(0, 1, 500))
    adf_res = analyzer.adf_test(stationary_s, "Stationary Series")
    assert adf_res.is_stationary is True

    # 2. Distribution Moments
    dist_res = analyzer.distribution_analysis(stationary_s, "Gaussian")
    assert dist_res.is_normal is True
    assert abs(dist_res.skewness) < 0.2
