"""
Unit Tests for Candle Resampler.
"""

import pytest
import pandas as pd
from microquant.pipeline.resampler import CandleResampler


def test_resampler_ohlc(sample_clean_trades):
    resampler = CandleResampler(time_frame="1min")
    candles = resampler.resample(sample_clean_trades)

    assert "open" in candles.columns
    assert "high" in candles.columns
    assert "low" in candles.columns
    assert "close" in candles.columns
    assert "volume" in candles.columns
    assert "trade_count" in candles.columns
    assert "vwap" in candles.columns

    # High should be >= Low, Open, and Close
    assert (candles["high"] >= candles["low"]).all()
    assert (candles["high"] >= candles["open"]).all()
    assert (candles["high"] >= candles["close"]).all()
    assert (candles["volume"] >= 0).all()
    assert (candles["trade_count"] >= 0).all()


def test_resampler_subminute_microbars(sample_clean_trades):
    resampler = CandleResampler(time_frame="5s")
    candles = resampler.resample(sample_clean_trades)

    # 60 ticks spanning 120s should generate multiple 5s candles
    assert len(candles) >= 10
    assert candles["volume"].sum() > 0
