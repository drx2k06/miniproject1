"""
Order Book and Trades Data Cleaning Pipeline.
Cleans raw, noisy, multi-format Level-3 market microstructure data.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import re
import numpy as np
import pandas as pd

from microquant.core.logging import get_logger
from microquant.core.exceptions import DataValidationError

logger = get_logger("microquant.cleaner")


@dataclass
class CleaningReport:
    """Summary of data cleaning metrics and transformations."""
    initial_rows: int
    final_rows: int
    duplicates_dropped: int
    null_rows_dropped: int
    non_positive_dropped: int
    crossed_books_dropped: int
    scale_anomalies_repaired: int
    outliers_dropped: int
    sides_imputed: int
    symbols_normalized: int
    time_span_start: str
    time_span_end: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OrderBookCleaner:
    """
    Production-grade market microstructure cleaning pipeline.
    Transforms raw, multi-format, corrupted orderbook & trade streams into clean datasets.
    """

    NUMERIC_COLS = [
        "price", "quantity",
        "bid_price_1", "bid_qty_1", "bid_price_2", "bid_qty_2", "bid_price_3", "bid_qty_3",
        "ask_price_1", "ask_qty_1", "ask_price_2", "ask_qty_2", "ask_price_3", "ask_qty_3",
    ]

    def __init__(self, max_deviation_from_mid: float = 0.02, apply_iqr_filter: bool = True):
        self.max_deviation_from_mid = max_deviation_from_mid
        self.apply_iqr_filter = apply_iqr_filter

    @staticmethod
    def _parse_timestamp(val: Any) -> pd.Timestamp:
        """Robustly parses timestamps across Unix epochs, slash dates, and ISO strings."""
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        # Unix epoch
        if s.isdigit():
            v = int(s)
            unit = "ms" if len(s) >= 13 else "s"
            return pd.to_datetime(v, unit=unit, utc=True)
        # European / Slash dates (e.g. DD/MM/YYYY)
        if "/" in s:
            return pd.to_datetime(s, dayfirst=True, utc=True)
        # Standard ISO or standard string
        return pd.to_datetime(s, utc=True)

    def clean(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Executes full multi-pass cleaning pipeline on raw trade and order book data.

        Args:
            df_raw: Raw DataFrame loaded from CSV or database.

        Returns:
            Tuple of (cleaned_df, cleaning_report).
        """
        logger.info(f"Starting data cleaning on {len(df_raw)} raw records...")
        initial_rows = len(df_raw)

        if initial_rows == 0:
            raise DataValidationError("Input dataframe is empty.")

        df = df_raw.copy()

        # Step 1: Remove exact duplicate rows
        initial_dups = df.duplicated().sum()
        df = df.drop_duplicates()
        logger.debug(f"Dropped {initial_dups} duplicate raw rows.")

        # Step 2: Normalize timestamps
        logger.debug("Normalizing heterogeneous timestamps to UTC...")
        df["timestamp"] = df["timestamp"].apply(self._parse_timestamp)
        df = df.dropna(subset=["timestamp"])

        # Step 3: Standardize symbols
        logger.debug("Standardizing instrument symbols...")
        sym_initial_unique = df["symbol"].nunique(dropna=False)
        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.upper()
            .str.replace("/", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.strip()
        )

        # Step 4: Clean all numeric columns (Level 1, Level 2, Level 3)
        logger.debug("Cleaning numeric currency representations ($ and commas)...")
        for col in self.NUMERIC_COLS:
            if col in df.columns:
                if df[col].dtype == object or df[col].dtype == "string":
                    cleaned_series = (
                        df[col]
                        .astype(str)
                        .str.replace(r"[$,\s]", "", regex=True)
                        .replace(["nan", "None", ""], np.nan)
                    )
                    df[col] = pd.to_numeric(cleaned_series, errors="coerce")
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Step 5: Clean and impute trade sides (buy / sell)
        logger.debug("Standardizing and imputing trade sides...")
        df["side"] = df["side"].astype(str).str.strip().str.lower()
        df.loc[~df["side"].isin(["buy", "sell"]), "side"] = np.nan

        missing_sides_before = df["side"].isna().sum()
        # Quote proximity rule
        at_ask = (df["price"] >= df["ask_price_1"]) & df["side"].isna()
        at_bid = (df["price"] <= df["bid_price_1"]) & df["side"].isna()
        df.loc[at_ask, "side"] = "buy"
        df.loc[at_bid, "side"] = "sell"

        # Tick rule fallback for remaining missing sides
        if df["side"].isna().any():
            price_diff = df["price"].diff()
            df.loc[df["side"].isna() & (price_diff > 0), "side"] = "buy"
            df.loc[df["side"].isna() & (price_diff < 0), "side"] = "sell"
            df["side"] = df["side"].bfill().ffill()

        sides_imputed = missing_sides_before - df["side"].isna().sum()

        # Step 6: Drop rows missing essential Level 1 fields
        essential_cols = ["price", "quantity", "bid_price_1", "bid_qty_1", "ask_price_1", "ask_qty_1", "side"]
        null_rows_before = len(df)
        df = df.dropna(subset=essential_cols)
        null_rows_dropped = null_rows_before - len(df)

        # Step 7: Filter non-positive prices and quantities
        non_pos_mask = (df["price"] <= 0) | (df["quantity"] <= 0) | (df["bid_price_1"] <= 0) | (df["ask_price_1"] <= 0)
        non_positive_dropped = non_pos_mask.sum()
        df = df[~non_pos_mask]

        # Step 8: Crossed book removal (Level 1 bid must be <= ask)
        crossed_mask = df["bid_price_1"] > df["ask_price_1"]
        crossed_books_dropped = crossed_mask.sum()
        df = df[~crossed_mask]

        # Step 9: Outlier repair & filtering
        mid_price = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
        gap = (df["price"] / mid_price - 1.0).abs()

        # Detect 100x pricing anomaly (e.g. 6210000 instead of 62100)
        scale_100x_error = (df["price"] / 100.0 / mid_price - 1.0).abs() < self.max_deviation_from_mid
        scale_anomalies_repaired = scale_100x_error.sum()
        df.loc[scale_100x_error, "price"] = df["price"] / 100.0

        # Recalculate gap and filter outliers exceeding max_deviation_from_mid
        gap_after = (df["price"] / mid_price - 1.0).abs()
        outliers_dropped = (gap_after > self.max_deviation_from_mid).sum()
        df = df[gap_after <= self.max_deviation_from_mid]

        # Step 10: Statistical IQR filter on price (if enabled)
        if self.apply_iqr_filter and len(df) > 10:
            q1 = df["price"].quantile(0.25)
            q3 = df["price"].quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            iqr_outliers = (df["price"] < lower_fence) | (df["price"] > upper_fence)
            logger.debug(f"Statistical IQR bounds [{lower_fence:.2f}, {upper_fence:.2f}]: {iqr_outliers.sum()} outliers.")
            df = df[~iqr_outliers]

        # Step 11: Sort chronologically and deduplicate trade_id
        df = df.sort_values("timestamp").reset_index(drop=True)
        if "trade_id" in df.columns:
            trade_id_dups = df["trade_id"].duplicated().sum()
            df = df.drop_duplicates(subset=["trade_id"]).reset_index(drop=True)
        else:
            trade_id_dups = 0

        # Build comprehensive audit report
        final_rows = len(df)
        time_span_start = str(df["timestamp"].min()) if final_rows > 0 else "N/A"
        time_span_end = str(df["timestamp"].max()) if final_rows > 0 else "N/A"

        report = CleaningReport(
            initial_rows=initial_rows,
            final_rows=final_rows,
            duplicates_dropped=int(initial_dups + trade_id_dups),
            null_rows_dropped=int(null_rows_dropped),
            non_positive_dropped=int(non_positive_dropped),
            crossed_books_dropped=int(crossed_books_dropped),
            scale_anomalies_repaired=int(scale_anomalies_repaired),
            outliers_dropped=int(outliers_dropped),
            sides_imputed=int(sides_imputed),
            symbols_normalized=int(sym_initial_unique),
            time_span_start=time_span_start,
            time_span_end=time_span_end,
        )

        logger.info(
            f"Cleaning completed: {initial_rows} -> {final_rows} rows retained "
            f"({final_rows / initial_rows * 100:.1f}% retention). Timespan: {time_span_start} to {time_span_end}."
        )
        return df, report
