"""
Market Microstructure and High-Frequency Feature Extraction.
Extracts depth-weighted orderbook signals, flow toxicity, and microstructure alphas.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from microquant.core.logging import get_logger
from microquant.core.exceptions import DataValidationError

logger = get_logger("microquant.features")


class MicrostructureFeatureExtractor:
    """
    Computes mathematical order book microstructure metrics and quantitative alpha signals.
    """

    CORE_FEATURE_COLUMNS: List[str] = [
        "spread",
        "spread_bps",
        "mid_price",
        "micro_price",
        "micro_price_bias",
        "order_imbalance_1",
        "depth_imbalance_3",
        "weighted_imbalance",
        "total_depth",
        "bid_slope",
        "ask_slope",
        "slope_ratio",
        "signed_quantity",
        "vwap",
        "price_to_vwap_bps",
        "effective_spread",
        "return_1",
        "log_return_1",
        "volatility_10",
        "volatility_20",
        "ma_10",
        "ma_30",
        "ma_ratio",
        "rsi_14",
        "bb_pct_b",
    ]

    def __init__(self, rsi_period: int = 14, vol_window: int = 20):
        self.rsi_period = rsi_period
        self.vol_window = vol_window

    def extract(self, df_clean: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """
        Derives all microstructure features from a cleaned trade and order book stream.

        Args:
            df_clean: Cleaned DataFrame with Level-1 to Level-3 columns.
            dropna: Whether to drop initial rolling window NaN rows.

        Returns:
            DataFrame augmented with quantitative microstructure features.
        """
        logger.info(f"Extracting microstructure features from {len(df_clean)} records...")
        if len(df_clean) < self.vol_window:
            raise DataValidationError(
                f"Dataframe has {len(df_clean)} rows, which is fewer than required window {self.vol_window}."
            )

        df = df_clean.copy()
        eps = 1e-9

        # 1. Level-1 Spread, Mid-Price, and Micro-Price
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
        df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
        df["spread_bps"] = (df["spread"] / df["mid_price"]) * 10000.0
        df["order_imbalance_1"] = (df["bid_qty_1"] - df["ask_qty_1"]) / (df["bid_qty_1"] + df["ask_qty_1"] + eps)

        # Micro-Price (Volume-Weighted Level-1 Mid Price)
        df["micro_price"] = (
            (df["bid_price_1"] * df["ask_qty_1"] + df["ask_price_1"] * df["bid_qty_1"])
            / (df["bid_qty_1"] + df["ask_qty_1"] + eps)
        )
        df["micro_price_bias"] = (df["micro_price"] - df["mid_price"]) / (df["spread"] + eps)

        # 2. Multi-Level Depth and Order Book Structure
        bid_cols_exist = all(c in df.columns for c in ["bid_price_2", "bid_qty_2", "bid_price_3", "bid_qty_3"])
        ask_cols_exist = all(c in df.columns for c in ["ask_price_2", "ask_qty_2", "ask_price_3", "ask_qty_3"])

        if bid_cols_exist and ask_cols_exist:
            total_bid_depth = df["bid_qty_1"] + df["bid_qty_2"].fillna(0) + df["bid_qty_3"].fillna(0)
            total_ask_depth = df["ask_qty_1"] + df["ask_qty_2"].fillna(0) + df["ask_qty_3"].fillna(0)
            df["total_depth"] = total_bid_depth + total_ask_depth
            df["depth_imbalance_3"] = (total_bid_depth - total_ask_depth) / (df["total_depth"] + eps)

            # Decaying depth weights (50% L1, 30% L2, 20% L3)
            w_bid = 0.5 * df["bid_qty_1"] + 0.3 * df["bid_qty_2"].fillna(0) + 0.2 * df["bid_qty_3"].fillna(0)
            w_ask = 0.5 * df["ask_qty_1"] + 0.3 * df["ask_qty_2"].fillna(0) + 0.2 * df["ask_qty_3"].fillna(0)
            df["weighted_imbalance"] = (w_bid - w_ask) / (w_bid + w_ask + eps)

            # Order book slopes (price delta over total depth)
            df["bid_slope"] = (df["bid_price_1"] - df["bid_price_3"].fillna(df["bid_price_1"])) / (total_bid_depth + eps)
            df["ask_slope"] = (df["ask_price_3"].fillna(df["ask_price_1"]) - df["ask_price_1"]) / (total_ask_depth + eps)
            df["slope_ratio"] = df["bid_slope"] / (df["ask_slope"] + eps)
        else:
            df["total_depth"] = df["bid_qty_1"] + df["ask_qty_1"]
            df["depth_imbalance_3"] = df["order_imbalance_1"]
            df["weighted_imbalance"] = df["order_imbalance_1"]
            df["bid_slope"] = 0.0
            df["ask_slope"] = 0.0
            df["slope_ratio"] = 1.0

        # 3. Trade Flow and Impact
        df["signed_quantity"] = np.where(df["side"] == "buy", df["quantity"], -df["quantity"])
        cum_volume = df["quantity"].cumsum()
        cum_pv = (df["price"] * df["quantity"]).cumsum()
        df["vwap"] = cum_pv / (cum_volume + eps)
        df["price_to_vwap_bps"] = (df["price"] - df["vwap"]) / (df["vwap"] + eps) * 10000.0
        df["effective_spread"] = 2.0 * (df["price"] - df["mid_price"]).abs()

        # 4. Returns and Volatility
        df["return_1"] = df["price"].pct_change()
        df["log_return_1"] = np.log(df["price"] / df["price"].shift(1))
        df["volatility_10"] = df["return_1"].rolling(10).std()
        df["volatility_20"] = df["return_1"].rolling(self.vol_window).std()

        # 5. Trend & Moving Averages
        df["ma_10"] = df["price"].rolling(10).mean()
        df["ma_30"] = df["price"].rolling(30).mean()
        df["ma_ratio"] = (df["ma_10"] / (df["ma_30"] + eps)) - 1.0

        # 6. Wilder's Exponentially Smoothed RSI
        delta = df["price"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + eps)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # 7. Bollinger Bands %B
        bb_mid = df["price"].rolling(20).mean()
        bb_std = df["price"].rolling(20).std()
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        df["bb_pct_b"] = (df["price"] - bb_lower) / (bb_upper - bb_lower + eps)

        if dropna:
            rows_before = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.info(f"Dropped {rows_before - len(df)} initial rolling NaN rows. Result: {len(df)} feature rows.")
        else:
            df = df.reset_index(drop=True)

        return df
