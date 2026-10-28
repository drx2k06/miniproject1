"""
Pytest Fixtures and Mock Data for MicroQuant Tests.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


@pytest.fixture
def sample_dirty_orderbook():
    """Returns a dirty DataFrame with common real-world corruptions."""
    data = {
        "trade_id": ["10001", "10002", "10003", "10004", "10005", "10006", "10007"],
        "timestamp": [
            "1790845352",                   # unix seconds
            "2026-10-01 09:00:05.123",       # standard iso
            "01/10/2026 09:00:10",           # slash format
            "2026-10-01 09:00:15.500",
            "2026-10-01 09:00:20.000",
            "2026-10-01 09:00:25.000",
            "2026-10-01 09:00:30.000",
        ],
        "symbol": ["BTCUSDT", "BTC/USDT", "btcusdt", "BTC-USDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
        "side": ["buy", "sell", "BUY", "Sell", None, "buy", "sell"],
        "price": ["$62,000.50", "62001.00", "62002.50", "62001.80", "62003.00", "6200300.00", "62002.00"],  # note 100x scale anomaly
        "quantity": ["1.250", "0.500", "2.100", "1.000", "0.800", "1.500", "0.900"],
        "bid_price_1": ["62000.0", "62000.5", "62001.0", "62001.0", "62002.0", "62002.5", "62001.5"],
        "bid_qty_1": ["2.0", "1.5", "3.0", "1.2", "2.5", "1.8", "2.0"],
        "bid_price_2": ["61999.0", "61999.5", "62000.0", "62000.0", "62001.0", "62001.5", "62000.5"],
        "bid_qty_2": ["3.0", "2.5", "4.0", "2.2", "3.5", "2.8", "3.0"],
        "bid_price_3": ["61998.0", "61998.5", "61999.0", "61999.0", "62000.0", "62000.5", "61999.5"],
        "bid_qty_3": ["4.0", "3.5", "5.0", "3.2", "4.5", "3.8", "4.0"],
        "ask_price_1": ["62001.0", "62001.5", "62002.0", "62002.0", "62003.0", "62003.5", "62002.5"],
        "ask_qty_1": ["1.8", "1.2", "2.5", "1.5", "2.0", "1.6", "1.9"],
        "ask_price_2": ["62002.0", "62002.5", "62003.0", "62003.0", "62004.0", "62004.5", "62003.5"],
        "ask_qty_2": ["2.8", "2.2", "3.5", "2.5", "3.0", "2.6", "2.9"],
        "ask_price_3": ["62003.0", "62003.5", "62004.0", "62004.0", "62005.0", "62005.5", "62004.5"],
        "ask_qty_3": ["3.8", "3.2", "4.5", "3.5", "4.0", "3.6", "3.9"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_clean_trades():
    """Generates synthetic sequence of 60 cleaned tick trades."""
    np.random.seed(42)
    n = 60
    base_time = pd.Timestamp("2026-10-01 09:00:00", tz="UTC")
    timestamps = [base_time + pd.Timedelta(seconds=i * 2) for i in range(n)]

    prices = 62000.0 + np.cumsum(np.random.normal(0, 0.5, n))
    quantities = np.random.uniform(0.1, 2.5, n)
    sides = np.random.choice(["buy", "sell"], n)

    df = pd.DataFrame({
        "trade_id": [str(100000 + i) for i in range(n)],
        "timestamp": timestamps,
        "symbol": ["BTCUSDT"] * n,
        "side": sides,
        "price": prices,
        "quantity": quantities,
        "bid_price_1": prices - 0.2,
        "bid_qty_1": np.random.uniform(1.0, 5.0, n),
        "bid_price_2": prices - 0.5,
        "bid_qty_2": np.random.uniform(2.0, 6.0, n),
        "bid_price_3": prices - 0.8,
        "bid_qty_3": np.random.uniform(3.0, 7.0, n),
        "ask_price_1": prices + 0.2,
        "ask_qty_1": np.random.uniform(1.0, 5.0, n),
        "ask_price_2": prices + 0.5,
        "ask_qty_2": np.random.uniform(2.0, 6.0, n),
        "ask_price_3": prices + 0.8,
        "ask_qty_3": np.random.uniform(3.0, 7.0, n),
    })
    return df
