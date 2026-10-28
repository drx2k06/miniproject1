"""
Unit Tests for Order Book Cleaning Pipeline.
"""

import pytest
import pandas as pd
from microquant.pipeline.cleaner import OrderBookCleaner


def test_cleaner_currency_and_symbol_standardization(sample_dirty_orderbook):
    cleaner = OrderBookCleaner()
    df_clean, report = cleaner.clean(sample_dirty_orderbook)

    # All symbols should be BTCUSDT
    assert (df_clean["symbol"] == "BTCUSDT").all()

    # Price and numeric columns should be floats without commas or dollar signs
    assert pd.api.types.is_float_dtype(df_clean["price"])
    assert pd.api.types.is_float_dtype(df_clean["bid_price_1"])
    assert pd.api.types.is_float_dtype(df_clean["ask_price_1"])


def test_cleaner_scale_anomaly_repair(sample_dirty_orderbook):
    cleaner = OrderBookCleaner()
    df_clean, report = cleaner.clean(sample_dirty_orderbook)

    # 100x outlier (6200300.0) should have been repaired to ~62003.0
    assert (df_clean["price"] < 100000.0).all()
    assert report.scale_anomalies_repaired >= 1


def test_cleaner_side_imputation(sample_dirty_orderbook):
    cleaner = OrderBookCleaner()
    df_clean, report = cleaner.clean(sample_dirty_orderbook)

    # Missing side should be imputed to buy or sell
    assert df_clean["side"].isna().sum() == 0
    assert set(df_clean["side"].unique()).issubset({"buy", "sell"})


def test_cleaner_crossed_books():
    # Construct crossed order book: bid > ask
    dirty_data = pd.DataFrame({
        "trade_id": ["1", "2"],
        "timestamp": ["2026-10-01 09:00:00", "2026-10-01 09:00:01"],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "side": ["buy", "sell"],
        "price": [62000.0, 62001.0],
        "quantity": [1.0, 1.0],
        "bid_price_1": [62005.0, 62000.0],  # row 1 is crossed (bid 62005 > ask 62001)
        "bid_qty_1": [1.0, 1.0],
        "ask_price_1": [62001.0, 62001.0],
        "ask_qty_1": [1.0, 1.0],
    })

    cleaner = OrderBookCleaner()
    df_clean, report = cleaner.clean(dirty_data)
    assert len(df_clean) == 1
    assert report.crossed_books_dropped == 1
    assert df_clean["trade_id"].iloc[0] == "2"
