"""
Unit Tests for Microstructure Feature Extraction.
"""

import pytest
import numpy as np
import pandas as pd
from microquant.pipeline.features import MicrostructureFeatureExtractor


def test_feature_extraction_columns(sample_clean_trades):
    fe = MicrostructureFeatureExtractor(rsi_period=14, vol_window=20)
    df_feat = fe.extract(sample_clean_trades, dropna=True)

    expected_cols = [
        "spread", "spread_bps", "mid_price", "micro_price",
        "order_imbalance_1", "depth_imbalance_3", "vwap", "rsi_14", "volatility_20"
    ]
    for col in expected_cols:
        assert col in df_feat.columns, f"Expected column {col} missing"

    # Mid price should be between best bid and best ask
    assert (df_feat["mid_price"] >= df_feat["bid_price_1"]).all()
    assert (df_feat["mid_price"] <= df_feat["ask_price_1"]).all()

    # Wilder's RSI should be bounded [0, 100]
    assert (df_feat["rsi_14"] >= 0.0).all()
    assert (df_feat["rsi_14"] <= 100.0).all()

    # Order imbalance should be bounded [-1, 1]
    assert (df_feat["order_imbalance_1"] >= -1.0).all()
    assert (df_feat["order_imbalance_1"] <= 1.0).all()


def test_micro_price_weighted():
    # Bid 100 (qty 10), Ask 102 (qty 2). Since ask qty is smaller, micro-price should skew towards ask!
    df = pd.DataFrame({
        "trade_id": [str(i) for i in range(25)],
        "timestamp": pd.date_range("2026-10-01", periods=25, freq="s", tz="UTC"),
        "symbol": ["BTCUSDT"] * 25,
        "side": ["buy"] * 25,
        "price": [101.0] * 25,
        "quantity": [1.0] * 25,
        "bid_price_1": [100.0] * 25,
        "bid_qty_1": [10.0] * 25,
        "ask_price_1": [102.0] * 25,
        "ask_qty_1": [2.0] * 25,
    })

    fe = MicrostructureFeatureExtractor()
    df_feat = fe.extract(df, dropna=False)

    # Micro-price = (100*2 + 102*10) / 12 = 1220 / 12 = 101.666
    expected_micro = (100.0 * 2.0 + 102.0 * 10.0) / 12.0
    assert np.isclose(df_feat["micro_price"].iloc[0], expected_micro)
    assert df_feat["micro_price"].iloc[0] > df_feat["mid_price"].iloc[0]
