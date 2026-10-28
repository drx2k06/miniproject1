"""
Multi-Resolution Time-Series OHLCV Resampling Engine.

Author: Daniel Richard Xavier
"""

from typing import Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OHLCVResampler:
    """Aggregates asynchronous trade ticks into uniform OHLCV candlesticks across arbitrary intervals."""

    def __init__(self, default_timeframe: str = "1min"):
        self.default_timeframe = default_timeframe

    def resample(
        self,
        df: pd.DataFrame,
        timeframe: Optional[str] = None,
        fill_gaps: bool = True
    ) -> pd.DataFrame:
        """
        Resamples a trade tick DataFrame into uniform OHLCV bars.

        Args:
            df: DataFrame containing 'timestamp', 'price', 'quantity', and 'side'.
            timeframe: Resampling frequency (e.g. '1s', '5s', '1min', '5min').
            fill_gaps: Whether to forward-fill quiet periods with 0 volume.

        Returns:
            DataFrame indexed by timestamp containing OHLCV and volume dynamics.
        """
        tf = timeframe or self.default_timeframe
        logger.info(f"Resampling {len(df)} tick records to '{tf}' resolution...")

        df_work = df.copy()
        if not isinstance(df_work.index, pd.DatetimeIndex):
            if "timestamp" not in df_work.columns:
                raise ValueError("DataFrame must contain a 'timestamp' column or DatetimeIndex.")
            df_work["timestamp"] = pd.to_datetime(df_work["timestamp"], utc=True)
            df_work = df_work.sort_values("timestamp").set_index("timestamp")

        # Core OHLC aggregation
        ohlc = df_work["price"].resample(tf).ohlc()

        # Volume and Trade Count aggregation
        ohlc["volume"] = df_work["quantity"].resample(tf).sum()
        ohlc["trade_count"] = df_work["price"].resample(tf).count()

        # Signed flow decomposition
        if "side" in df_work.columns:
            buy_mask = df_work["side"].str.lower() == "buy"
            ohlc["buy_volume"] = df_work.loc[buy_mask, "quantity"].resample(tf).sum()
            ohlc["sell_volume"] = df_work.loc[~buy_mask, "quantity"].resample(tf).sum()
        else:
            ohlc["buy_volume"] = np.nan
            ohlc["sell_volume"] = np.nan

        # Bar VWAP
        df_work["pv"] = df_work["price"] * df_work["quantity"]
        pv_sum = df_work["pv"].resample(tf).sum()
        vol_sum = df_work["quantity"].resample(tf).sum()
        ohlc["vwap"] = pv_sum / np.where(vol_sum == 0, np.nan, vol_sum)

        if fill_gaps:
            ohlc["close"] = ohlc["close"].ffill()
            ohlc["open"] = ohlc["open"].fillna(ohlc["close"])
            ohlc["high"] = ohlc["high"].fillna(ohlc["close"])
            ohlc["low"] = ohlc["low"].fillna(ohlc["close"])
            ohlc["volume"] = ohlc["volume"].fillna(0.0)
            ohlc["trade_count"] = ohlc["trade_count"].fillna(0)
            ohlc["buy_volume"] = ohlc["buy_volume"].fillna(0.0)
            ohlc["sell_volume"] = ohlc["sell_volume"].fillna(0.0)
            ohlc["vwap"] = ohlc["vwap"].fillna(ohlc["close"])

        ohlc["return"] = ohlc["close"].pct_change()
        ohlc["log_return"] = np.log(ohlc["close"] / ohlc["close"].shift(1))
        ohlc["hl_ratio"] = (ohlc["high"] - ohlc["low"]) / ohlc["close"]

        # Reset index to have 'timestamp' as a column
        res = ohlc.reset_index()
        logger.info(f"Constructed {len(res)} candles at '{tf}' granularity.")
        return res
