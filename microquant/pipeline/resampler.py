"""
Multi-Resolution Candlestick Resampler.
Aggregates tick streams into standard 1-minute candles and sub-minute micro-bars.
"""

from typing import Optional
import numpy as np
import pandas as pd

from microquant.core.logging import get_logger
from microquant.core.exceptions import DataValidationError

logger = get_logger("microquant.resampler")


class CandleResampler:
    """
    Transforms tick-level trade data into structured OHLCV time bars with microstructural volume metrics.
    """

    def __init__(self, time_frame: str = "1min"):
        self.time_frame = time_frame

    def resample(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples cleaned tick data into time-series candles.

        Args:
            df_clean: Cleaned DataFrame with timestamp, price, quantity, side.

        Returns:
            DataFrame indexed by timestamp with OHLCV and volume flow metrics.
        """
        logger.info(f"Resampling {len(df_clean)} ticks into '{self.time_frame}' candles...")
        if "timestamp" not in df_clean.columns or "price" not in df_clean.columns:
            raise DataValidationError("Input dataframe missing 'timestamp' or 'price'.")

        df = df_clean.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
            except Exception:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        df = df.sort_values("timestamp").set_index("timestamp")

        # Core OHLC
        ohlc = df["price"].resample(self.time_frame).ohlc()

        # Volume and Trade Count
        ohlc["volume"] = df["quantity"].resample(self.time_frame).sum()
        ohlc["trade_count"] = df["price"].resample(self.time_frame).count()

        # Buy Volume vs Sell Volume
        if "side" in df.columns:
            buy_vol = df.loc[df["side"] == "buy", "quantity"].resample(self.time_frame).sum()
            sell_vol = df.loc[df["side"] == "sell", "quantity"].resample(self.time_frame).sum()
            ohlc["buy_volume"] = buy_vol
            ohlc["sell_volume"] = sell_vol
            ohlc["order_flow_imbalance"] = (ohlc["buy_volume"] - ohlc["sell_volume"]) / (ohlc["volume"] + 1e-9)
        else:
            ohlc["buy_volume"] = 0.0
            ohlc["sell_volume"] = 0.0
            ohlc["order_flow_imbalance"] = 0.0

        # Bar VWAP
        pv = (df["price"] * df["quantity"]).resample(self.time_frame).sum()
        ohlc["vwap"] = pv / (ohlc["volume"] + 1e-9)

        # Average Spread if available
        if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
            spread = (df["ask_price_1"] - df["bid_price_1"]).resample(self.time_frame).mean()
            ohlc["avg_spread"] = spread

        # Forward fill empty intervals
        ohlc["close"] = ohlc["close"].ffill()
        ohlc["open"] = ohlc["open"].fillna(ohlc["close"])
        ohlc["high"] = ohlc["high"].fillna(ohlc["close"])
        ohlc["low"] = ohlc["low"].fillna(ohlc["close"])
        ohlc["vwap"] = ohlc["vwap"].fillna(ohlc["close"])
        ohlc["volume"] = ohlc["volume"].fillna(0.0)
        ohlc["trade_count"] = ohlc["trade_count"].fillna(0).astype(int)
        ohlc["buy_volume"] = ohlc["buy_volume"].fillna(0.0)
        ohlc["sell_volume"] = ohlc["sell_volume"].fillna(0.0)
        ohlc["order_flow_imbalance"] = ohlc["order_flow_imbalance"].fillna(0.0)

        # Reset index to have timestamp column
        candles = ohlc.reset_index()
        logger.info(f"Generated {len(candles)} '{self.time_frame}' candles.")
        return candles
