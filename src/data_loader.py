"""
Robust Data Loading and Schema Validation Module.

Author: Daniel Richard Xavier
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when dataset fails critical schema or integrity validation."""
    pass


@dataclass
class ValidationReport:
    """Summary of data schema, dimensions, nulls, and duplicate counts."""
    total_rows: int
    total_columns: int
    columns: List[str]
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    duplicate_rows: int
    duplicate_trade_ids: int
    is_valid: bool
    validation_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "columns": self.columns,
            "missing_counts": self.missing_counts,
            "missing_percentages": self.missing_percentages,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_trade_ids": self.duplicate_trade_ids,
            "is_valid": self.is_valid,
            "validation_messages": self.validation_messages,
        }


class DataLoader:
    """Loads, inspects, and validates high-frequency market microstructure data."""

    EXPECTED_RAW_COLUMNS = [
        "trade_id", "timestamp", "symbol", "side", "price", "quantity",
        "bid_price_1", "bid_qty_1", "bid_price_2", "bid_qty_2",
        "bid_price_3", "bid_qty_3", "ask_price_1", "ask_qty_1",
        "ask_price_2", "ask_qty_2", "ask_price_3", "ask_qty_3"
    ]

    def __init__(self, raw_path: Optional[Union[str, Path]] = None):
        self.raw_path = Path(raw_path) if raw_path else None

    def load_raw_data(self, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Loads the raw CSV dataset safely preserving all string representations.
        
        Args:
            filepath: Optional override path for the raw CSV.
            
        Returns:
            pd.DataFrame: Raw records loaded as strings.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            DataValidationError: If the loaded dataset is empty or corrupted.
        """
        target_path = Path(filepath) if filepath else self.raw_path
        if not target_path:
            raise ValueError("No file path specified for DataLoader.")

        if not target_path.exists():
            raise FileNotFoundError(f"Raw data file not found at: {target_path}")

        logger.info(f"Loading raw dataset from {target_path}...")
        try:
            df = pd.read_csv(target_path, dtype=str)
        except Exception as e:
            raise DataValidationError(f"Failed to read CSV file: {e}") from e

        if df.empty:
            raise DataValidationError(f"Dataset at {target_path} is completely empty.")

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns.")
        return df

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        allow_extra: bool = True
    ) -> ValidationReport:
        """
        Performs comprehensive schema and dimension checks on a dataframe.
        
        Args:
            df: The dataframe to inspect.
            required_columns: Columns required to be present.
            allow_extra: Whether additional columns are allowed.
            
        Returns:
            ValidationReport with diagnostics.
        """
        if required_columns is None:
            required_columns = self.EXPECTED_RAW_COLUMNS

        messages: List[str] = []
        is_valid = True

        actual_cols = list(df.columns)
        missing_required = set(required_columns) - set(actual_cols)
        if missing_required:
            messages.append(f"Missing required columns: {sorted(missing_required)}")
            is_valid = False

        if not allow_extra:
            unexpected = set(actual_cols) - set(required_columns)
            if unexpected:
                messages.append(f"Unexpected extra columns: {sorted(unexpected)}")
                is_valid = False

        total_rows = int(len(df))
        null_counts = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
        null_pcts = {str(k): round(float(v / max(total_rows, 1)) * 100, 2) for k, v in null_counts.items()}

        dup_rows = int(df.duplicated().sum())
        dup_ids = int(df["trade_id"].duplicated().sum()) if "trade_id" in df.columns else 0

        if total_rows == 0:
            messages.append("Dataframe contains 0 records.")
            is_valid = False

        report = ValidationReport(
            total_rows=total_rows,
            total_columns=len(actual_cols),
            columns=actual_cols,
            missing_counts=null_counts,
            missing_percentages=null_pcts,
            duplicate_rows=dup_rows,
            duplicate_trade_ids=dup_ids,
            is_valid=is_valid,
            validation_messages=messages,
        )
        return report

    def save_dataset(self, df: pd.DataFrame, filepath: Union[str, Path], index: bool = False) -> Path:
        """Saves a dataframe to CSV with directory creation and logging."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=index)
        logger.info(f"Successfully saved {len(df)} records to {out_path}.")
        return out_path
