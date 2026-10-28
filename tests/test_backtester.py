"""
Unit Tests for Microstructure Strategy Backtester.
"""

import pytest
import numpy as np
import pandas as pd
from microquant.models.backtester import MicrostructureBacktester


def test_backtester_metrics():
    n = 100
    np.random.seed(42)
    prices = 62000.0 + np.cumsum(np.random.normal(0, 1.0, n))
    probs = np.random.uniform(0.3, 0.7, n)

    bt = MicrostructureBacktester(
        maker_fee_bps=1.0,
        taker_fee_bps=4.0,
        slippage_bps=1.5,
        initial_capital=100000.0,
        long_threshold=0.52,
        short_threshold=0.48,
    )
    metrics, history = bt.run_backtest(probabilities=probs, prices=prices)

    assert metrics.initial_capital == 100000.0
    assert metrics.final_equity > 0
    assert len(history) == n
    assert "equity" in history.columns
    assert "benchmark_equity" in history.columns
    assert "drawdown" in history.columns
    assert metrics.max_drawdown_pct <= 0.0  # drawdown is non-positive percentage
