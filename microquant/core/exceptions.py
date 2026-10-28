"""
Custom Exception Classes for MicroQuant Platform.
"""

class MicroQuantError(Exception):
    """Base exception for all MicroQuant platform errors."""
    pass


class DataValidationError(MicroQuantError):
    """Raised when data fails schema, format, or invariant validations."""
    pass


class PipelineExecutionError(MicroQuantError):
    """Raised when an error occurs during ETL, feature extraction, or resampling."""
    pass


class ModelTrainingError(MicroQuantError):
    """Raised when model training, scaling, or hyperparameter optimization fails."""
    pass


class InferenceError(MicroQuantError):
    """Raised when real-time or batch prediction inference fails."""
    pass


class BacktestExecutionError(MicroQuantError):
    """Raised when quantitative strategy backtesting fails."""
    pass
