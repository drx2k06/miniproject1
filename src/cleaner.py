"""
High-Frequency Market Microstructure Data Cleaning & Sanitization Engine.

Author: Daniel Richard Xavier
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import logging
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningAudit:
    """Tracks before-and-after audit statistics across the data cleaning pipeline."""
    initial_records: int
    duplicate_rows_removed: int
    timestamps_parsed: int
    symbols_standardized: int
    side_imputations_quote_rule: int
    side_imputations_tick_rule: int
    non_positive_values_dropped: int
    crossed_books_dropped: int
    scale_anomalies_corrected: int
    extreme_deviations_dropped: int
    null_quotes_dropped: int
    final_clean_records: int
    retention_rate_pct: float

    def to_dict(self) -> Dict:
        return asdict(self)


class DataCleaner:
    """Sanitizes dirty trade and order book tick feeds into clean research data."""

    NUMERIC_COLUMNS = [
        "price", "quantity",
        "bid_price_1", "bid_qty_1", "bid_price_2", "bid_qty_2", "bid_price_3", "bid_qty_3",
        "ask_price_1", "ask_qty_1", "ask_price_2", "ask_qty_2", "ask_price_3", "ask_qty_3"
    ]

    def __init__(self, max_mid_gap_pct: float = 0.02):
        self.max_mid_gap_pct = max_mid_gap_pct

    @staticmethod
    def parse_mixed_timestamp(val: object) -> pd.Timestamp:
        """
        Parses heterogeneous timestamps: Unix epoch seconds/ms, UK DD/MM/YYYY, or ISO-8601.
        Returns a UTC-localized pandas Timestamp.
        """
        if pd.isna(val):
            return pd.NaT

        s = str(val).strip()

        # 1. Pure numeric strings -> Unix epoch
        if s.isdigit():
            # If length >= 13, it is milliseconds; otherwise seconds
            unit = "ms" if len(s) >= 13 else "s"
            return pd.to_datetime(int(s), unit=unit, utc=True)

        # 2. Date containing slashes -> DD/MM/YYYY format
        if "/" in s:
            return pd.to_datetime(s, dayfirst=True, utc=True)

        # 3. Standard ISO-8601 datetime
        return pd.to_datetime(s, utc=True)

    def clean(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningAudit]:
        """
        Executes end-to-end cleaning pipeline on the raw dataset.
        
        Args:
            df_raw: Raw DataFrame loaded from dirty CSV.
            
        Returns:
            Tuple[pd.DataFrame, CleaningAudit]: Cleaned DataFrame and audit record.
        """
        initial_count = len(df_raw)
        logger.info(f"Initiating cleaning pipeline on {initial_count} raw records...")

        # 1. Drop exact duplicate rows
        df = df_raw.drop_duplicates().copy()
        dup_rows_removed = initial_count - len(df)

        # 2. Parse timestamps to UTC datetime
        df["timestamp"] = df["timestamp"].apply(self.parse_mixed_timestamp)
        timestamps_parsed = df["timestamp"].notna().sum()

        # 3. Standardize symbol representations (e.g., BTC/USDT -> BTCUSDT)
        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.upper()
            .str.replace("/", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.strip()
        )
        symbols_standardized = len(df)

        # 4. Standardize side casing
        df["side"] = df["side"].astype(str).str.strip().str.lower()
        df.loc[df["side"].isin(["nan", "none", ""], ), "side"] = np.nan

        # 5. Sanitize numeric fields (strip currency symbols and commas)
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[$,]", "", regex=True),
                    errors="coerce"
                )

        # 6. Impute missing trade direction (Lee-Ready Quote Rule & Tick Rule)
        # Quote Rule: trade at or above ask is buyer-initiated; at or below bid is seller-initiated
        initial_missing_side = df["side"].isna().sum()
        
        quote_buy_mask = df["side"].isna() & (df["price"] >= df["ask_price_1"])
        quote_sell_mask = df["side"].isna() & (df["price"] <= df["bid_price_1"])
        df.loc[quote_buy_mask, "side"] = "buy"
        df.loc[quote_sell_mask, "side"] = "sell"
        quote_imputed = int(quote_buy_mask.sum() + quote_sell_mask.sum())

        # Tick Rule fallback: if still missing, compare to previous valid trade price
        still_missing = df["side"].isna()
        tick_imputed = 0
        if still_missing.any():
            price_diff = df["price"].diff()
            df.loc[still_missing & (price_diff >= 0), "side"] = "buy"
            df.loc[still_missing & (price_diff < 0), "side"] = "sell"
            # Any remaining un-imputed default to 'buy'
            df["side"] = df["side"].fillna("buy")
            tick_imputed = int(still_missing.sum())

        # 7. Drop rows with null essential Level-1 fields
        essential_cols = ["price", "quantity", "bid_price_1", "bid_qty_1", "ask_price_1", "ask_qty_1", "side"]
        before_null_drop = len(df)
        df = df.dropna(subset=essential_cols).copy()
        null_quotes_dropped = before_null_drop - len(df)

        # 8. Filter non-positive values
        before_nonpos = len(df)
        df = df[(df["price"] > 0) & (df["quantity"] > 0) & (df["bid_price_1"] > 0) & (df["ask_price_1"] > 0)].copy()
        nonpos_dropped = before_nonpos - len(df)

        # 9. Filter crossed books (Bid >= Ask violates matching engine integrity)
        before_crossed = len(df)
        df = df[df["bid_price_1"] < df["ask_price_1"]].copy()
        crossed_dropped = before_crossed - len(df)

        # 10. Correct decimal scale multiplier errors (100x price scale corruption)
        mid_quote = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
        scale_err_mask = ((df["price"] / 100.0 / mid_quote - 1.0).abs() < 0.05)
        scale_anomalies_corrected = int(scale_err_mask.sum())
        df.loc[scale_err_mask, "price"] = df.loc[scale_err_mask, "price"] / 100.0

        # 11. Remove extreme price outliers relative to mid-price
        mid_quote = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
        rel_gap = (df["price"] / mid_quote - 1.0).abs()
        before_gap = len(df)
        df = df[rel_gap <= self.max_mid_gap_pct].copy()
        extreme_dropped = before_gap - len(df)

        # 12. Chronological sorting and unique trade_id enforcement
        df = df.sort_values("timestamp").drop_duplicates(subset=["trade_id"]).reset_index(drop=True)
        final_count = len(df)
        retention_pct = round((final_count / initial_count) * 100, 2)

        audit = CleaningAudit(
            initial_records=int(initial_count),
            duplicate_rows_removed=int(dup_rows_removed),
            timestamps_parsed=int(timestamps_parsed),
            symbols_standardized=int(symbols_standardized),
            side_imputations_quote_rule=int(quote_imputed),
            side_imputations_tick_rule=int(tick_imputed),
            non_positive_values_dropped=int(nonpos_dropped),
            crossed_books_dropped=int(crossed_dropped),
            scale_anomalies_corrected=int(scale_anomalies_corrected),
            extreme_deviations_dropped=int(extreme_dropped),
            null_quotes_dropped=int(null_quotes_dropped),
            final_clean_records=int(final_count),
            retention_rate_pct=float(retention_pct),
        )

        logger.info(
            f"Cleaning completed: {final_count}/{initial_count} records retained ({retention_pct}%)."
        )
        return df, audit
